from typing import Any, Dict, Iterator, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env
from pydantic import Field, SecretStr

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
        SystemMessage,
    )
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    raise ImportError(
        "Could not import azure-ai-inference library. "
        "Please install it with `pip install azure-ai-inference`."
    )


from langchain_core.language_models.chat_models import (
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessageChunk,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGenerationChunk,
)
from langchain_core.utils import convert_to_secret_str


def _create_message_dicts(messages: List[BaseMessage]) -> List[Dict]:
    messages_dicts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            messages_dicts.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            messages_dicts.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages_dicts.append({"role": "assistant", "content": msg.content})
        else:
            raise ValueError(f"Got unknown type {msg}")
    return messages_dicts


class GithubLLM(BaseChatModel):
    """
    GitHub Copilot chat model through Azure Inference endpoint.

    Setup:
        To use, you should have the environment variable ``GITHUB_TOKEN`` set
        with your GitHub token.

    Key init args — client params:
        github_token: Your GitHub token.

    Key init args — completion params:
        endpoint: The Azure Inference endpoint. Defaults to
            "https://models.inference.ai.azure.com".
        model_name: The name of the GitHub model to use. Defaults to "gpt-4o".
        temperature: Model temperature. Defaults to 1.0.
        top_p: Model top_p. Defaults to 1.0.
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
        streaming: Whether to stream the response. Defaults to False.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import GithubLLM

            llm = GithubLLM(github_token="YOUR_GITHUB_TOKEN")

    Invoke:
        .. code-block:: python
            messages = [
                SystemMessage(content="your are an AI assistant."),
                HumanMessage(content="tell me a joke."),
            ]
            response = llm.invoke(messages)
    """

    github_token: SecretStr = Field(default="")  # type: ignore
    """Your GitHub token."""

    endpoint: str = Field(default="https://models.inference.ai.azure.com")
    """The Azure Inference endpoint."""

    model_name: str = Field(default="gpt-4o")
    """The name of the GitHub model to use."""

    temperature: float = Field(default=1.0)
    """Model temperature."""

    top_p: float = Field(default=1.0)
    """Model top_p."""

    max_tokens: int = Field(default=1000)
    """Maximum number of tokens to generate."""

    streaming: bool = Field(default=False)
    """Whether to stream the response."""

    class Config:
        """Configuration for this pydantic object."""

        populate_by_name = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"github_token": "GITHUB_TOKEN"}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "endpoint": self.endpoint,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "github-llm"

    def __init__(self, **kwargs: Any):
        kwargs["github_token"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "github_token", "GITHUB_TOKEN")
        )
        super().__init__(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.github_token.get_secret_value()),
        )

        messages_azure = [
            {"role": m.type, "content": m.content} for m in messages
        ]  # Convert to Azure format.

        # if there is 'human' in the role we change it to 'user'
        for message in messages_azure:
            if message["role"] == "human":
                message["role"] = "user"

        params = {
            "messages": messages_azure,
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": stop,
        }

        if self.streaming:
            params["stream"] = True
            response = client.complete(**params)
            return generate_from_stream(
                self._stream_response(response, run_manager)
            )

        response = client.complete(**params)

        message = AIMessage(
            content=response.choices[0].message.content,
            additional_kwargs={},
            # remove metadata reference
        )
        generation = ChatGeneration(message=message)
        client.close()
        return ChatResult(generations=[generation])

    def _stream_response(
        self,
        response,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content or ""
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=content)
                )
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    llm = GithubLLM(github_token=os.getenv("GITHUB_TOKEN"), model_name="gpt-4o")
    # simple test
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg)
