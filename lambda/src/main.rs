use anyhow::{Context, Result};
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use text_embeddings_backend::{DType, ModelType};
use text_embeddings_core::download::download_artifacts;
use text_embeddings_core::infer::Infer;
use text_embeddings_core::queue::Queue;
use text_embeddings_core::tokenization::Tokenization;
use tokenizers::decoders::metaspace::PrependScheme;
use tokenizers::{PreTokenizerWrapper, Tokenizer};

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    pub pad_token_id: usize,
    pub id2label: Option<HashMap<String, String>>,
    pub label2id: Option<HashMap<String, usize>>,
}

#[derive(Deserialize)]
struct Request {
    inputs: Vec<String>,
}

#[derive(Serialize)]
pub(crate) struct EmbedResponse(Vec<Vec<f32>>);

async fn handler(infer: &Infer, event: LambdaEvent<Request>) -> Result<EmbedResponse, Error> {
    let input = event.payload.inputs[0].clone();

    // do embed
    let truncate = false;
    let normalize = true;

    // let input = "hello world".to_string();
    let permit = infer
        .try_acquire_permit()
        .context("Could not acquire permit")?;

    let response = infer.embed(input, truncate, normalize, permit).await?;

    Ok(EmbedResponse(vec![response.results]))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let infer = setup_infer().await.context("Could not setup infer")?;
    let shared_infer = &infer;
    lambda_runtime::run(service_fn(move |event: LambdaEvent<Request>| async move {
        handler(&shared_infer, event).await
    }))
    .await
}

async fn setup_infer() -> Result<Infer> {
    let model_id = "jegormeister/robbert-v2-dutch-base-mqa-finetuned";

    let api = ApiBuilder::new().with_progress(true).build().unwrap();

    let api_repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let model_root = download_artifacts(&api_repo)
        .await
        .context("Could not download model artifacts")?;

    println!("{}", model_root.to_str().unwrap());

    // Load config
    let config_path = model_root.join("config.json");
    let config = fs::read_to_string(config_path).context("`config.json` not found")?;
    let config: ModelConfig =
        serde_json::from_str(&config).context("Failed to parse `config.json`")?;

    let pool = text_embeddings_backend::Pool::Mean;
    let model_type = ModelType::Embedding(pool);

    let tokenizer_path = model_root.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect(
        "tokenizer.json not found. text-embeddings-inference only supports fast tokenizers",
    );

    // See https://github.com/huggingface/tokenizers/pull/1357
    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() {
        if let PreTokenizerWrapper::Metaspace(m) = pre_tokenizer {
            // We are forced to clone since `Tokenizer` does not have a `get_mut` for `pre_tokenizer`
            let mut m = m.clone();
            m.set_prepend_scheme(PrependScheme::First);
            tokenizer.with_pre_tokenizer(PreTokenizerWrapper::Metaspace(m));
        } else if let PreTokenizerWrapper::Sequence(s) = pre_tokenizer {
            // We are forced to clone since `Tokenizer` does not have a `get_mut` for `pre_tokenizer`
            let mut s = s.clone();
            for pre_tokenizer in s.get_pre_tokenizers_mut() {
                if let PreTokenizerWrapper::Metaspace(m) = pre_tokenizer {
                    m.set_prepend_scheme(PrependScheme::First);
                }
            }
            tokenizer.with_pre_tokenizer(PreTokenizerWrapper::Sequence(s));
        }
    }

    tokenizer.with_padding(None);

    // Position IDs offset. Used for Roberta and camembert.
    let position_offset = if &config.model_type == "xlm-roberta"
        || &config.model_type == "camembert"
        || &config.model_type == "roberta"
    {
        config.pad_token_id + 1
    } else {
        0
    };
    let max_input_length = config.max_position_embeddings - position_offset;

    let tokenization_workers = num_cpus::get_physical();

    // Tokenization logic
    let tokenization = Tokenization::new(
        tokenization_workers,
        tokenizer,
        max_input_length,
        position_offset,
    );

    let dtype = DType::Float32;

    let backend = text_embeddings_backend::Backend::new(
        model_root,
        dtype,
        model_type,
        "/tmp/text-embeddings-inference-server".to_string(),
        None,
    )
    .context("Could not create backend")?;

    backend
        .health()
        .await
        .context("Model backend is not healthy")?;

    // Queue logic
    let max_batch_requests = backend.max_batch_size;
    let max_batch_tokens: usize = 16384;
    let max_concurrent_requests: usize = 512;
    let queue = Queue::new(
        max_batch_tokens,
        max_batch_requests,
        max_concurrent_requests,
    );

    // Create infer task
    let infer = Infer::new(tokenization, queue, max_concurrent_requests, backend);

    println!("health: {}", infer.health().await);

    return Ok(infer);
}
