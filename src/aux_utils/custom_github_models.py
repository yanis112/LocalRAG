from openai import OpenAI
from pydantic import Field, SecretStr
from typing import Dict, List, Any, Optional
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.utils import from_env, secret_from_env
from typing_extensions import Self
# We ignore the "unused imports" here since we want to reexport these from this package.
from pydantic import (
    model_validator,
)

GITHUB_BASE_URL = "https://models.inference.ai.azure.com"

class ChatGithubModels(BaseChatOpenAI):
    r"""ChatGithubModels chat model.
    This model is a wrapper around the OpenAI API for the Chat endpoint.
    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"github_api_key": "GITHUB_TOKEN"}
        """
        return {"github_api_key": "GITHUB_TOKEN"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "github"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        attributes: Dict[str, Any] = {}

        if self.github_api_base:
            attributes["github_api_base"] = self.github_api_base

        if self.github_proxy:
            attributes["github_proxy"] = self.github_proxy

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "github-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "github"
        return params

    model_name: str = Field(alias="model")
    """Model name to use."""
    github_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("GITHUB_TOKEN", default=None),
    )
    """Automatically inferred from env are `GITHUB_TOKEN` if not provided."""
    github_api_base: str = Field(
        default_factory=from_env("GITHUB_API_BASE", default=GITHUB_BASE_URL),
        alias="base_url",
    )

    github_proxy: str = Field(default_factory=from_env("GITHUB_PROXY", default=""))

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params = {
            "api_key": (
                self.github_api_key.get_secret_value()
                if self.github_api_key
                else None
            ),
            # Ensure we always fallback to the GitHub API url.
            "base_url": self.github_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        if self.github_proxy and (self.http_client or self.http_async_client):
            raise ValueError(
                "Cannot specify 'github_proxy' if one of "
                "'http_client'/'http_async_client' is already specified. Received:\n"
                f"{self.github_proxy=}\n{self.http_client=}\n{self.http_async_client=}"
            )
        if not self.client:
            if self.github_proxy and not self.http_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_client = httpx.Client(proxy=self.github_proxy)
            sync_specific = {"http_client": self.http_client}
            self.root_client = OpenAI(**client_params, **sync_specific)  # type: ignore
            self.client = self.root_client.chat.completions
        if not self.async_client:
            if self.github_proxy and not self.http_async_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_async_client = httpx.AsyncClient(proxy=self.github_proxy)
            async_specific = {"http_client": self.http_async_client}
            self.root_async_client = OpenAI(
                **client_params,  # type: ignore
                **async_specific,  # type: ignore
            )
            self.async_client = self.root_async_client.chat.completions
        return self

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    chat = ChatGithubModels(
        model="gpt-4o",
        github_api_key="GITHUB_TOKEN",
        github_api_base="https://api.github.com",
    )
    
    # Call the chat model
    response = chat("What is the capital of France?")
    print(response)