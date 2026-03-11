from collections.abc import Iterable

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from cadence_md.app.settings import MODEL_INFERENCE_API_KEY, MODEL_INFERENCE_BASE_URL, RAGConfig


class LLMWrapper:
    """
    A wrapper to preserve the .invoke(prompt: str) -> str interface
    instead of ChatOpenAI's AIMessage.
    """

    def __init__(self, chat_model: ChatOpenAI):
        self.chat_model = chat_model

    def invoke(self, prompt: str) -> str:
        """ """
        resp = self.chat_model.invoke(prompt)
        # ChatOpenAI returns an AIMessage, we take the text content
        return resp.content

    def stream(self, prompt: str) -> Iterable[str]:
        """ """
        # ChatOpenAI ожидает список сообщений[web:110][web:124]
        messages = [HumanMessage(content=prompt)]
        for chunk in self.chat_model.stream(messages):
            # chunk – AIMessageChunk, content может быть None[web:124][web:125]
            if chunk.content:
                yield chunk.content


def get_llm(config: RAGConfig, streaming: bool = False):
    """
    An LLM that runs on an LM Studio OpenAI-compatible server.
    An LLM model named config.llm.model_name must be running in inference server.
    """

    llm = ChatOpenAI(
        model=config.llm.model_name,
        base_url=MODEL_INFERENCE_BASE_URL,
        api_key=MODEL_INFERENCE_API_KEY,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_new_tokens,
        top_p=config.llm.top_p,
        streaming=streaming,
    )

    return LLMWrapper(llm)
