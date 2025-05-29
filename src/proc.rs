use ollama_rs::Ollama;
use ollama_rs::generation::completion::GenerationResponse;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::embeddings::GenerateEmbeddingsResponse;
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::models::{LocalModel, ModelInfo};
use prosa::core::adaptor::Adaptor;
use prosa::core::error::ProcError;
use prosa::core::msg::{InternalMsg, Msg};
use prosa::core::proc::{Proc, ProcBusParam, proc, proc_settings};
use prosa::core::service::ServiceError;
use serde::Deserialize;
use std::str::FromStr;
use thiserror::Error;
use tracing::info;
use url::Url;

use crate::adaptor::OllamaAdaptor;

#[derive(Debug, Error)]
/// ProSA Ollama error
pub enum OllamaError {
    /// IO error
    #[error("Ollama error `{0}`")]
    Ollama(#[from] ollama_rs::error::OllamaError),
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
            OllamaError::Other(error) => ServiceError::UnableToReachService(error),
        }
    }
}

impl ProcError for OllamaError {
    fn recoverable(&self) -> bool {
        match self {
            OllamaError::Ollama(_error) => false,
            OllamaError::Other(_error) => false,
        }
    }
}

#[proc_settings]
#[derive(Debug, Deserialize)]
pub struct OllamaProcSettings {
    /// Url of the Ollama API server
    #[serde(default = "OllamaProcSettings::default_url")]
    url: Url,
    /// List of model that will be used with the processor
    #[serde(default)]
    models: Vec<String>,
    /// Allow insecure connections to the library. Only use this if you are pulling from your own library during development.
    allow_insecure: bool,
    /// Service declared for the processor
    service: String,
}

impl OllamaProcSettings {
    fn default_url() -> Url {
        Url::from_str("http://localhost:11434").unwrap()
    }

    /// Create a settings with Ollama URL and processor service name
    pub fn new(url: Url, allow_insecure: bool, service: String) -> OllamaProcSettings {
        OllamaProcSettings {
            url,
            models: Vec::new(),
            allow_insecure,
            service,
            ..Default::default()
        }
    }

    /// Setter to list model that processor need before starting
    pub fn set_models(&mut self, models: Vec<String>) {
        self.models = models;
    }

    pub fn get_ollama(&self) -> Ollama {
        Ollama::from_url(self.url.clone())
    }
}

#[proc_settings]
impl Default for OllamaProcSettings {
    fn default() -> Self {
        OllamaProcSettings {
            url: Self::default_url(),
            models: Vec::default(),
            allow_insecure: false,
            service: String::from("ollama"),
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
        let ollama = self.settings.get_ollama();

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
            .add_service_proc(vec![self.settings.service.clone()])
            .await?;

        loop {
            if let Some(msg) = self.internal_rx_queue.recv().await {
                match msg {
                    InternalMsg::Request(msg) => {
                        let ollama_request =
                            adaptor.process_request(msg.get_service(), msg.get_data());
                        match ollama_request {
                            Ok(OllamaRequest::ListLocalModels) => {
                                match ollama.list_local_models().await {
                                    Ok(local_model) => {
                                        match adaptor.process_ollama_response(
                                            OllamaResponse::LocalModels(local_model),
                                        ) {
                                            Ok(resp) => msg.return_to_sender(resp).await?,
                                            Err(e) => {
                                                msg.return_error_to_sender(None, e.into()).await?
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        msg.return_error_to_sender(
                                            None,
                                            OllamaError::Ollama(e).into(),
                                        )
                                        .await?
                                    }
                                }
                            }
                            Ok(OllamaRequest::ModelInfo(model_name)) => {
                                match ollama.show_model_info(model_name).await {
                                    Ok(model_info) => {
                                        match adaptor.process_ollama_response(
                                            OllamaResponse::ModelInfo(model_info),
                                        ) {
                                            Ok(resp) => msg.return_to_sender(resp).await?,
                                            Err(e) => {
                                                msg.return_error_to_sender(None, e.into()).await?
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        msg.return_error_to_sender(
                                            None,
                                            OllamaError::Ollama(e).into(),
                                        )
                                        .await?
                                    }
                                }
                            }
                            Ok(OllamaRequest::GenerateRequest(request)) => {
                                match ollama.generate(*request).await {
                                    Ok(response) => {
                                        match adaptor.process_ollama_response(response.into()) {
                                            Ok(resp) => msg.return_to_sender(resp).await?,
                                            Err(e) => {
                                                msg.return_error_to_sender(None, e.into()).await?
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        msg.return_error_to_sender(
                                            None,
                                            OllamaError::Ollama(e).into(),
                                        )
                                        .await?
                                    }
                                }
                            }
                            Ok(OllamaRequest::GenerateEmbeddingsRequest(embeddings_request)) => {
                                match ollama.generate_embeddings(*embeddings_request).await {
                                    Ok(response) => {
                                        match adaptor.process_ollama_response(response.into()) {
                                            Ok(resp) => msg.return_to_sender(resp).await?,
                                            Err(e) => {
                                                msg.return_error_to_sender(None, e.into()).await?
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        msg.return_error_to_sender(
                                            None,
                                            OllamaError::Ollama(e).into(),
                                        )
                                        .await?
                                    }
                                }
                            }
                            Err(e) => msg.return_error_to_sender(None, e.into()).await?,
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
