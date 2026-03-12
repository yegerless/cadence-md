from collections.abc import Iterable

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from cadence_md.app.settings import settings


class LLMWrapper:
    """ """

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


def get_llm(streaming: bool = False):
    """ """

    llm = ChatOpenAI(
        model=settings.rag_config.llm.model_name,
        base_url=settings.MODEL_INFERENCE_BASE_URL,
        api_key=settings.MODEL_INFERENCE_API_KEY,  # type: ignore
        temperature=settings.rag_config.llm.temperature,
        max_completion_tokens=settings.rag_config.llm.max_new_tokens,
        top_p=settings.rag_config.llm.top_p,
        streaming=streaming,
    )

    return LLMWrapper(llm)
