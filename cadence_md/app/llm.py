from collections.abc import Iterable

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class LLMWrapper:
    """
    Wrapper for the ChatOpenAI model

    Args:
        chat_model: The ChatOpenAI model to wrap.
    Returns:
        LLMWrapper: A wrapper for the ChatOpenAI model.
    """

    def __init__(self, chat_model: ChatOpenAI):
        self.chat_model = chat_model

    def invoke(self, prompt: str) -> str:
        """
        Invoke the ChatOpenAI model.

        Args:
            prompt: The prompt to invoke the model with.
        Returns:
            str: The response from the model.
        """
        resp = self.chat_model.invoke(prompt)
        # ChatOpenAI returns an AIMessage, we take the text content
        return resp.content

    def stream(self, prompt: str) -> Iterable[str]:
        """
        Stream the response from the ChatOpenAI model.

        Args:
            prompt: The prompt to stream the response from.
        Returns:
            Iterable[str]: A generator of the response from the model.
        """
        # ChatOpenAI expects a list of messages
        messages = [HumanMessage(content=prompt)]
        for chunk in self.chat_model.stream(messages):
            # chunk – AIMessageChunk, content may be None
            if chunk.content:
                yield chunk.content


def get_llm(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_completion_tokens: int,
    top_p: float,
    streaming: bool,
) -> LLMWrapper:
    """
    Get a LLMWrapper instance.

    Args:
        model: The model name to use.
        base_url: The base URL of the model inference server.
        api_key: The API key to use.
        temperature: The temperature to use.
        max_completion_tokens: The maximum completion tokens to use.
        top_p: The top p to use.
        streaming: Whether to stream the output.
    Returns:
        LLMWrapper: An instance of the LLMWrapper class.
    """
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        streaming=streaming,
    )

    return LLMWrapper(llm)
