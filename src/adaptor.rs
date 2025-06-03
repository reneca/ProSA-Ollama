use prosa::core::msg::RequestMsg;

use crate::proc::{OllamaError, OllamaProc, OllamaRequest, OllamaResponse};

pub trait OllamaAdaptor<M>
where
    M: 'static
        + std::marker::Send
        + std::marker::Sync
        + std::marker::Sized
        + std::clone::Clone
        + std::fmt::Debug
        + prosa_utils::msg::tvf::Tvf
        + std::default::Default,
{
    /// Method called when the processor spawns to create a new adaptor
    /// This method is called only once so the processing will be thread safe
    fn new(proc: &OllamaProc<M>) -> Result<Self, OllamaError>
    where
        Self: Sized;

    /// Method to process incomming requests
    fn process_request<'a>(
        &mut self,
        service_name: &str,
        request: &M,
    ) -> Result<OllamaRequest<'a>, OllamaError>;

    /// Method to process Ollama responses
    fn process_ollama_response(
        &mut self,
        response: OllamaResponse,
        original_request: &RequestMsg<M>,
    ) -> Result<M, OllamaError>;
}
