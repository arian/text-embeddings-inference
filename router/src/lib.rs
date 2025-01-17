/// Text Embedding Inference Webserver
pub mod server;

use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fmt::Formatter;
use text_embeddings_core::tokenization::EncodingInput;
use utoipa::openapi::{RefOr, Schema};
use utoipa::ToSchema;

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct EmbeddingModel {
    #[schema(example = "cls")]
    pub pooling: String,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ClassifierModel {
    #[schema(example = json!({"0": "LABEL"}))]
    pub id2label: HashMap<String, String>,
    #[schema(example = json!({"LABEL": "0"}))]
    pub label2id: HashMap<String, usize>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Classifier(ClassifierModel),
    Embedding(EmbeddingModel),
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Info {
    /// Model info
    #[schema(example = "thenlper/gte-base")]
    pub model_id: String,
    #[schema(nullable = true, example = "fca14538aa9956a46526bd1d0d11d69e19b5a101")]
    pub model_sha: Option<String>,
    #[schema(example = "float16")]
    pub model_dtype: String,
    pub model_type: ModelType,
    /// Router Parameters
    #[schema(example = "128")]
    pub max_concurrent_requests: usize,
    #[schema(example = "512")]
    pub max_input_length: usize,
    #[schema(example = "2048")]
    pub max_batch_tokens: usize,
    #[schema(nullable = true, example = "null", default = "null")]
    pub max_batch_requests: Option<usize>,
    #[schema(example = "32")]
    pub max_client_batch_size: usize,
    #[schema(example = "4")]
    pub tokenization_workers: usize,
    /// Router Info
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
}

#[derive(Debug)]
pub(crate) enum Sequence {
    Single(String),
    Pair(String, String),
}

impl Sequence {
    pub(crate) fn count_chars(&self) -> usize {
        match self {
            Sequence::Single(s) => s.chars().count(),
            Sequence::Pair(s1, s2) => s1.chars().count() + s2.chars().count(),
        }
    }
}

impl From<Sequence> for EncodingInput {
    fn from(value: Sequence) -> Self {
        match value {
            Sequence::Single(s) => Self::Single(s),
            Sequence::Pair(s1, s2) => Self::Dual(s1, s2),
        }
    }
}

#[derive(Debug)]
pub(crate) enum PredictInput {
    Single(Sequence),
    Batch(Vec<Sequence>),
}

impl<'de> Deserialize<'de> for PredictInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Internal {
            Single(String),
            Multiple(Vec<String>),
        }

        struct PredictInputVisitor;

        impl<'de> Visitor<'de> for PredictInputVisitor {
            type Value = PredictInput;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str(
                    "a string, \
                    a pair of strings [string, string] \
                    or a batch of mixed strings and pairs [[string], [string, string], ...]",
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(PredictInput::Single(Sequence::Single(v.to_string())))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let sequence_from_vec = |mut value: Vec<String>| {
                    // Validate that value is correct
                    match value.len() {
                        1 => Ok(Sequence::Single(value.pop().unwrap())),
                        2 => {
                            // Second element is last
                            let second = value.pop().unwrap();
                            let first = value.pop().unwrap();
                            Ok(Sequence::Pair(first, second))
                        }
                        // Sequence can only be a single string or a pair of strings
                        _ => Err(de::Error::invalid_length(value.len(), &self)),
                    }
                };

                // Get first element
                // This will determine if input is a batch or not
                let s = match seq
                    .next_element::<Internal>()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?
                {
                    // Input is not a batch
                    // Return early
                    Internal::Single(value) => {
                        // Option get second element
                        let second = seq.next_element()?;

                        if seq.next_element::<String>()?.is_some() {
                            // Error as we do not accept > 2 elements
                            return Err(de::Error::invalid_length(3, &self));
                        }

                        if let Some(second) = second {
                            // Second element exists
                            // This is a pair
                            return Ok(PredictInput::Single(Sequence::Pair(value, second)));
                        } else {
                            // Second element does not exist
                            return Ok(PredictInput::Single(Sequence::Single(value)));
                        }
                    }
                    // Input is a batch
                    Internal::Multiple(value) => sequence_from_vec(value),
                }?;

                let mut batch = Vec::with_capacity(32);
                // Push first sequence
                batch.push(s);

                // Iterate on all sequences
                while let Some(value) = seq.next_element::<Vec<String>>()? {
                    // Validate sequence
                    let s = sequence_from_vec(value)?;
                    // Push to batch
                    batch.push(s);
                }
                Ok(PredictInput::Batch(batch))
            }
        }

        deserializer.deserialize_any(PredictInputVisitor)
    }
}

impl<'__s> ToSchema<'__s> for PredictInput {
    fn schema() -> (&'__s str, RefOr<Schema>) {
        (
            "PredictInput",
            utoipa::openapi::OneOfBuilder::new()
                .item(
                    utoipa::openapi::ObjectBuilder::new()
                        .schema_type(utoipa::openapi::SchemaType::String)
                        .description(Some("A single string")),
                )
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(
                            utoipa::openapi::ObjectBuilder::new()
                                .schema_type(utoipa::openapi::SchemaType::String),
                        )
                        .description(Some("A pair of strings"))
                        .min_items(Some(2))
                        .max_items(Some(2)),
                )
                .item(
                    utoipa::openapi::ArrayBuilder::new().items(
                        utoipa::openapi::OneOfBuilder::new()
                            .item(
                                utoipa::openapi::ArrayBuilder::new()
                                    .items(
                                        utoipa::openapi::ObjectBuilder::new()
                                            .schema_type(utoipa::openapi::SchemaType::String),
                                    )
                                    .description(Some("A single string"))
                                    .min_items(Some(1))
                                    .max_items(Some(1)),
                            )
                            .item(
                                utoipa::openapi::ArrayBuilder::new()
                                    .items(
                                        utoipa::openapi::ObjectBuilder::new()
                                            .schema_type(utoipa::openapi::SchemaType::String),
                                    )
                                    .description(Some("A pair of strings"))
                                    .min_items(Some(2))
                                    .max_items(Some(2)),
                            )
                    ).description(Some("A batch")),
                )
                .description(Some(
                    "Model input. \
                Can be either a single string, a pair of strings or a batch of mixed single and pairs \
                of strings.",
                ))
                .example(Some(json!("What is Deep Learning?")))
                .into(),
        )
    }
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct PredictRequest {
    pub inputs: PredictInput,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub truncate: bool,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Prediction {
    #[schema(example = "0.5")]
    score: f32,
    #[schema(example = "admiration")]
    label: String,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum PredictResponse {
    Single(Vec<Prediction>),
    Batch(Vec<Vec<Prediction>>),
}

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum Input {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct OpenAICompatRequest {
    pub input: Input,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    model: Option<String>,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    user: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatEmbedding {
    #[schema(example = "embedding")]
    object: &'static str,
    #[schema(example = json!(["0.0", "1.0", "2.0"]))]
    embedding: Vec<f32>,
    #[schema(example = "0")]
    index: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatUsage {
    #[schema(example = "512")]
    prompt_tokens: usize,
    #[schema(example = "512")]
    total_tokens: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatResponse {
    #[schema(example = "list")]
    object: &'static str,
    data: Vec<OpenAICompatEmbedding>,
    #[schema(example = "thenlper/gte-base")]
    model: String,
    usage: OpenAICompatUsage,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedRequest {
    pub inputs: Input,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub truncate: bool,
    #[serde(default = "default_normalize")]
    #[schema(default = "true", example = "true")]
    pub normalize: bool,
}

fn default_normalize() -> bool {
    true
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([["0.0", "1.0", "2.0"]]))]
pub(crate) struct EmbedResponse(Vec<Vec<f32>>);

#[derive(Serialize, ToSchema)]
pub(crate) enum ErrorType {
    Unhealthy,
    Backend,
    Overloaded,
    Validation,
    Tokenizer,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorResponse {
    pub error: String,
    pub error_type: ErrorType,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatErrorResponse {
    pub message: String,
    pub code: u16,
    #[serde(rename(serialize = "type"))]
    pub error_type: ErrorType,
}
