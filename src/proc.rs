use base64::{
    Engine as _,
    engine::general_purpose::STANDARD,
};
use ollama_rs::generation::chat::ChatMessageResponse;
use ollama_rs::headers::{HeaderMap, HeaderValue, InvalidHeaderValue};
use ollama_rs::Ollama;
use ollama_rs::generation::completion::GenerationResponse;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::embeddings::GenerateEmbeddingsResponse;
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::models::{LocalModel, ModelInfo};
use opentelemetry::KeyValue;
use prosa::core::adaptor::Adaptor;
use prosa::core::error::ProcError;
use prosa::core::msg::{InternalMsg, Msg};
use prosa::core::proc::{proc, proc_settings, Proc, ProcBusParam, ProcConfig as _};
use prosa::core::service::ServiceError;
use serde::{Deserialize, Serialize};
use std::env;
use std::str::FromStr;
use thiserror::Error;
use tracing::{debug, info, warn};
use url::Url;

use crate::adaptor::OllamaAdaptor;

#[derive(Debug, Error)]
/// ProSA Ollama error
pub enum OllamaError {
    /// IO error
    #[error("Ollama error `{0}`")]
    Ollama(#[from] ollama_rs::error::OllamaError),
    /// Header value error
    #[error("Invalide header value `{0}`")]
    InvalidHeaderValue(#[from] InvalidHeaderValue),
    /// Other error
    #[error("Ollama other error `{0}`")]
    Other(String),
}

impl From<OllamaError> for ServiceError {
    fn from(e: OllamaError) -> Self {
        match e {
            OllamaError::Ollama(ollama_error) => {
                ServiceError::UnableToReachService(ollama_error.to_string())
            }
            OllamaError::InvalidHeaderValue(e) => ServiceError::ProtocolError(e.to_string()),
            OllamaError::Other(error) => ServiceError::UnableToReachService(error),
        }
    }
}

impl ProcError for OllamaError {
    fn recoverable(&self) -> bool {
        match self {
            OllamaError::Ollama(_error) => false,
            OllamaError::InvalidHeaderValue(_error) => false,
            OllamaError::Other(_error) => false,
        }
    }
}

#[proc_settings]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaProcSettings {
    /// Url of the Ollama API server
    #[serde(default = "OllamaProcSettings::default_url")]
    url: Url,
    /// List of model that will be used with the processor
    #[serde(default)]
    models: Vec<String>,
    /// Allow insecure connections to the library. Only use this if you are pulling from your own library during development.
    #[serde(default)]
    allow_insecure: bool,
    /// Service declared for the processor
    #[serde(default = "OllamaProcSettings::default_services")]
    services: Vec<String>,
}

impl OllamaProcSettings {
    fn default_url() -> Url {
        env::var("OLLAMA_HOST")
            .or(env::var("OLLAMA_URL"))
            .ok()
            .and_then(|host| Url::from_str(&host).ok())
            .unwrap_or(Url::from_str("http://localhost:11434").unwrap())
    }

    fn default_services() -> Vec<String> {
        vec![String::from("ollama")]
    }

    /// Create a settings with Ollama URL and processor services names
    pub fn new(url: Url, allow_insecure: bool, services: Vec<String>) -> OllamaProcSettings {
        OllamaProcSettings {
            url,
            models: Vec::new(),
            allow_insecure,
            services,
            ..Default::default()
        }
    }

    /// Setter to list model that processor need before starting
    pub fn set_models(&mut self, models: Vec<String>) {
        self.models = models;
    }

    pub fn get_ollama(&self) -> Result<Ollama, OllamaError> {
        let mut ollama = Ollama::from_url(self.url.clone());
        let mut header_map = HeaderMap::new();

        if let Some(password) = self.url.password() {
            if self.url.username().is_empty() {
                header_map.insert("Authorization", HeaderValue::from_str(format!("Bearer {}", password).as_str())?);
            } else {
                header_map.insert("Authorization", HeaderValue::from_str(format!("Basic {}", STANDARD.encode(format!("{}:{}", self.url.username(), password))).as_str())?);
            }
        }

        if !header_map.is_empty() {
            ollama.set_headers(Some(header_map));
        }

        Ok(ollama)
    }
}

#[proc_settings]
impl Default for OllamaProcSettings {
    fn default() -> Self {
        OllamaProcSettings {
            url: Self::default_url(),
            models: Vec::default(),
            allow_insecure: false,
            services: Self::default_services(),
        }
    }
}

/// Ollama requests
pub enum OllamaRequest<'a> {
    ListLocalModels,
    ModelInfo(String),
    GenerateRequest(Box<GenerationRequest<'a>>),
    GenerateEmbeddingsRequest(Box<GenerateEmbeddingsRequest>),
}

/// Ollama responses
pub enum OllamaResponse {
    LocalModels(Vec<LocalModel>),
    ModelInfo(ModelInfo),
    GenerateResponse(Box<GenerationResponse>),
    GenerateEmbeddingsResponse(Box<GenerateEmbeddingsResponse>),
    ChatMessageResponse(Box<ChatMessageResponse>),
}

impl From<GenerationResponse> for OllamaResponse {
    fn from(response: GenerationResponse) -> Self {
        OllamaResponse::GenerateResponse(Box::new(response))
    }
}

impl From<GenerateEmbeddingsResponse> for OllamaResponse {
    fn from(response: GenerateEmbeddingsResponse) -> Self {
        OllamaResponse::GenerateEmbeddingsResponse(Box::new(response))
    }
}

impl From<ChatMessageResponse> for OllamaResponse {
    fn from(response: ChatMessageResponse) -> Self {
        OllamaResponse::ChatMessageResponse(Box::new(response))
    }
}

#[proc(settings = OllamaProcSettings)]
pub struct OllamaProc {}

// You must implement the trait Proc to define your processing
#[proc]
impl<A> Proc<A> for OllamaProc
where
    A: Default + Adaptor + OllamaAdaptor<M> + std::marker::Send + std::marker::Sync,
{
    async fn internal_run(
        &mut self,
        _name: String,
    ) -> Result<(), Box<dyn ProcError + Send + Sync>> {
        let ollama = self.settings.get_ollama()?;

        // List of models
        let local_models = ollama
            .list_local_models()
            .await
            .map_err(OllamaError::Ollama)?;

        // Pull missing models
        'model: for model in &self.settings.models {
            for local_model in &local_models {
                if &local_model.name == model {
                    continue 'model;
                }
            }

            // Pull model
            let pull_model_status = ollama
                .pull_model(model.to_string(), self.settings.allow_insecure)
                .await
                .map_err(OllamaError::Ollama)?;
            info!("Pulled the model {}: {:?}", model, pull_model_status);
        }

        // Initiate an adaptor for the Ollama processor
        let mut adaptor = A::new(self)?;

        // Declare the processor
        self.proc.add_proc().await?;

        // Add all service to listen
        self.proc
            .add_service_proc(self.settings.services.clone())
            .await?;

        // Meter to log AI statistics
        let meter = self.get_proc_param().meter("ollama");
        let observable_prompt_call_counter = meter
            .u64_counter("prosa_ollama_prompt_token_count")
            .with_description("Counter of prompt tokens")
            .build();
        let observable_gen_call_counter = meter
            .u64_counter("prosa_ollama_gen_token_count")
            .with_description("Counter of generated tokens")
            .build();
        let observable_token_histogram = meter
            .u64_histogram("prosa_ollama_token_histogram")
            .with_description("Histogram generations")
            .build();

        loop {
            if let Some(msg) = self.internal_rx_queue.recv().await {
                match msg {
                    InternalMsg::Request(mut msg) => {
                        if let Some(data) = msg.take_data() {
                            let enter_span = msg.enter_span();
                            let ollama_request = adaptor.process_request(msg.get_service(), data);
                            match ollama_request {
                                Ok(OllamaRequest::ListLocalModels) => {
                                    debug!("List local models");
                                    match ollama.list_local_models().await {
                                        Ok(local_model) => {
                                            match adaptor.process_ollama_response(
                                                OllamaResponse::LocalModels(local_model),
                                            ) {
                                                Ok(resp) => {
                                                    drop(enter_span);
                                                    msg.return_to_sender(resp).await?
                                                }
                                                Err(e) => {
                                                    drop(enter_span);
                                                    msg.return_error_to_sender(None, e.into())
                                                        .await?
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            drop(enter_span);
                                            msg.return_error_to_sender(
                                                None,
                                                OllamaError::Ollama(e).into(),
                                            )
                                            .await?
                                        }
                                    }
                                }
                                Ok(OllamaRequest::ModelInfo(model_name)) => {
                                    debug!("Model info {model_name}");
                                    match ollama.show_model_info(model_name).await {
                                        Ok(model_info) => {
                                            match adaptor.process_ollama_response(
                                                OllamaResponse::ModelInfo(model_info),
                                            ) {
                                                Ok(resp) => {
                                                    drop(enter_span);
                                                    msg.return_to_sender(resp).await?
                                                }
                                                Err(e) => {
                                                    drop(enter_span);
                                                    msg.return_error_to_sender(None, e.into())
                                                        .await?
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            drop(enter_span);
                                            msg.return_error_to_sender(
                                                None,
                                                OllamaError::Ollama(e).into(),
                                            )
                                            .await?
                                        }
                                    }
                                }
                                Ok(OllamaRequest::GenerateRequest(request)) => {
                                    debug!("Generate");
                                    match ollama.generate(*request).await {
                                        Ok(response) => {
                                            if let Some(prompt_eval_count) = response.prompt_eval_count {
                                                observable_prompt_call_counter.add(prompt_eval_count, &[
                                                    KeyValue::new("type", "gen"),
                                                    KeyValue::new("model", response.model.clone()),
                                                ]);
                                            }
                                            if let Some(eval_count) = response.eval_count {
                                                observable_gen_call_counter.add(eval_count, &[
                                                    KeyValue::new("type", "gen"),
                                                    KeyValue::new("model", response.model.clone()),
                                                ]);
                                            }
                                            if let Some(total_duration) = response.total_duration {
                                                observable_token_histogram.record(total_duration / 1000000, &[
                                                    KeyValue::new("type", "total"),
                                                    KeyValue::new("model", response.model.clone()),
                                                ]);
                                            }
                                            if let Some(load_duration) = response.load_duration {
                                                observable_token_histogram.record(load_duration / 1000000, &[
                                                    KeyValue::new("type", "load"),
                                                    KeyValue::new("model", response.model.clone()),
                                                ]);
                                            }
                                            if let Some(prompt_eval_duration) = response.prompt_eval_duration {
                                                observable_token_histogram.record(prompt_eval_duration / 1000000, &[
                                                    KeyValue::new("type", "prompt"),
                                                    KeyValue::new("model", response.model.clone()),
                                                ]);
                                            }
                                            if let Some(eval_duration) = response.eval_duration {
                                                observable_token_histogram.record(eval_duration / 1000000, &[
                                                    KeyValue::new("type", "eval"),
                                                    KeyValue::new("model", response.model.clone()),
                                                ]);
                                            }

                                            match adaptor.process_ollama_response(response.into()) {
                                                Ok(resp) => {
                                                    drop(enter_span);
                                                    msg.return_to_sender(resp).await?
                                                }
                                                Err(e) => {
                                                    drop(enter_span);
                                                    msg.return_error_to_sender(None, e.into())
                                                        .await?
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            drop(enter_span);
                                            msg.return_error_to_sender(
                                                None,
                                                OllamaError::Ollama(e).into(),
                                            )
                                            .await?
                                        }
                                    }
                                }
                                Ok(OllamaRequest::GenerateEmbeddingsRequest(
                                    embeddings_request,
                                )) => {
                                    debug!("Generate embeddings");
                                    match ollama.generate_embeddings(*embeddings_request).await {
                                        Ok(response) => {
                                            observable_gen_call_counter.add(response.embeddings.iter().len() as u64, &[
                                                KeyValue::new("type", "embed"),
                                            ]);

                                            match adaptor.process_ollama_response(response.into()) {
                                                Ok(resp) => {
                                                    drop(enter_span);
                                                    msg.return_to_sender(resp).await?
                                                }
                                                Err(e) => {
                                                    drop(enter_span);
                                                    msg.return_error_to_sender(None, e.into()).await?
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            drop(enter_span);
                                            msg.return_error_to_sender(
                                                None,
                                                OllamaError::Ollama(e).into(),
                                            )
                                            .await?
                                        }
                                    }
                                },
                                Err(e) => {
                                    warn!("Request error: {e}");
                                    drop(enter_span);
                                    msg.return_error_to_sender(None, e.into()).await?
                                }
                            }
                        }
                    }
                    InternalMsg::Response(msg) => panic!(
                        "The Ollama processor {} receive a response {:?}",
                        self.get_proc_id(),
                        msg
                    ),
                    InternalMsg::Error(err) => panic!(
                        "The Ollama processor {} receive an error {:?}",
                        self.get_proc_id(),
                        err
                    ),
                    InternalMsg::Command(_) => todo!(),
                    InternalMsg::Config => todo!(),
                    InternalMsg::Service(table) => self.service = table,
                    InternalMsg::Shutdown => {
                        adaptor.terminate();
                        self.proc.remove_proc(None).await?;
                        return Ok(());
                    }
                }
            }
        }
    }
}
