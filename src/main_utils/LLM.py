import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.structured import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import List, Union, Callable, Dict, Any

# Load environment variables
load_dotenv()

class CustomChatModel:
    """
    Represents a custom chat model.

    This class handles the initialization of different chat models based on the provided provider.
    It does NOT handle caching directly. The caching is done outside in the `load_chat_model` function.

    Args:
        model_name (str): The name of the language model.
        llm_provider (str): The provider of the language model.
        temperature (float, optional): The temperature for text generation. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
        top_k (int, optional): The top-k value for text generation. Defaults to 1.
        top_p (float, optional): The top-p value for text generation. Defaults to 0.01.
        system_prompt (str, optional): A custom system prompt. Defaults to "You are a helpful assistant."
    """
    def __init__(self, model_name, llm_provider, temperature=1.0, max_tokens=20000,top_k=45,top_p=0.95,system_prompt=None):
        self.model_name = model_name
        self.chat_model = None
        self.chat_prompt_template = None
        self.llm_provider = llm_provider

        self.system_prompt = system_prompt if system_prompt is not None else "You are a helpful assistant."

        # parameters
        self.llm_temperature = temperature  # controls the randomness of the output
        self.max_tokens = max_tokens  # maximum number of tokens to generate
        self.top_k = top_k  # controls the diversity of the output
        self.top_p = top_p  # controls the diversity of the output
        self.context_window_size = 0


        # Load model based on the provider
        self._load_llm()



    def _load_llm(self):
      """Loads the specific chat model based on the provider."""
      if self.llm_provider == "groq":
          from langchain_groq import ChatGroq
          self.context_window_size = 131072
          self.chat_model = ChatGroq(
              temperature=self.llm_temperature,
              model_name=self.model_name,
              groq_api_key=os.getenv("GROQ_API_KEY"),
              max_tokens=self.max_tokens,)
        
          self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{text}"),
                ]
            )


      elif self.llm_provider == "sambanova":
            from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
            os.environ["SAMBANOVA_API_KEY"] = os.getenv("SAMBANOVA_API_KEY")
            self.context_window_size = 8000
            self.chat_model = ChatSambaNovaCloud(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.llm_temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", "{text}"),
                ]
            )


      elif self.llm_provider == "github":
          from src.aux_utils.github_llm import GithubLLM
          self.chat_model = GithubLLM(
              github_token=os.getenv("GITHUB_TOKEN"),
              model_name=self.model_name,
              temperature=self.llm_temperature,
              top_p=self.top_p,
              top_k=self.top_k,
              max_tokens=self.max_tokens,
          )
          self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", "{text}"),
                ]
            )


      elif self.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold,HarmCategory

            self.chat_model = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.llm_temperature,
                max_tokens=self.max_tokens,
                timeout=None,
                max_retries=2,
                safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
                top_k=self.top_k,
                top_p=self.top_p,
            )
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", "{text}"),
                ]
            )

      elif self.llm_provider == "ollama":
          import ollama
          from langchain_ollama import ChatOllama
          template = ollama.show(self.model_name)["template"]
          self.context_window_size = 8192
          self.chat_model = ChatOllama(
              model=self.model_name,
              keep_alive=0,
              num_ctx=self.context_window_size,
              temperature=self.llm_temperature,
              template=template,
          )
          self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", "{text}"),
                ]
            )

      elif self.llm_provider == "cerebras":
          from langchain_cerebras import ChatCerebras

          #use load_dotenv() to load the environment variables
          os.environ["CEREBRAS_API_KEY"] = os.getenv("CEREBRAS_API_KEY")

          self.llm = ChatCerebras(
              model=self.model_name,
          )
          self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{text}"),
                ]
            )

      # elif self.llm_provider == "huggingface":
      #     # Find all available cuda devices
      #     import torch
      #     import transformers
      #     pipeline = transformers.pipeline(
      #         "text-generation",
      #         model=self.llm_name,
      #         model_kwargs={"torch_dtype": torch.bfloat16},
      #         device_map="auto",
      #     )

      #     terminators = [
      #         pipeline.tokenizer.eos_token_id,
      #         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
      #     ]

      #     self.chat_model = HuggingFacePipeline.from_model_id(
      #         model_id=self.llm_name,
      #         task="text-generation",
      #         device_map="auto",
      #         pipeline_kwargs=dict(
      #             max_new_tokens=1024,
      #             temperature=self.llm_temperature,
      #             torch_dtype=torch.bfloat16,
      #             eos_token_id=terminators,
      #             do_sample=True,
      #             top_p=0.9,
      #         ),
      #     )

      #     self.chat_model = ChatHuggingFace(llm=self.chat_model)

      #     self.chat_prompt_template = ChatPromptTemplate.from_messages(
      #         [
      #             ("system", self.system_prompt),
      #             ("human", "{text}"),
      #         ]
      #     )

      else:
        raise ValueError(
            "CRITICAL ERROR: THE LLM MODEL NAME IS NOT RECOGNIZED. VERIFY THE LLM NAME (OR LLM PROVIDER NAME) !"
        )

    def bind_tools(self, tool_list: List[Union[StructuredTool, Callable, Dict[str, Any]]]):
        """Binds tools to the chat model.

        Args:
            tool_list (list): A list of tools to bind to the chat model.
        Returns:
            CustomChatModel: A new CustomChatModel instance with bound tools
        """
        if not tool_list:
            return self  # No tools provided, return self

        if self.llm_provider == "groq":
           formatted_tools = []
           for tool in tool_list:
              if isinstance(tool, StructuredTool):
                formatted_tools.append(convert_to_openai_function(tool.func))
              elif isinstance(tool, dict) or callable(tool):
                formatted_tools.append(convert_to_openai_function(tool))
              else:
                raise ValueError("Unsupported tool type for Groq, please provide a structured tool, a function, or a dict")

           new_chat_model = self.chat_model.bind_tools(formatted_tools)
           new_instance = CustomChatModel(
                model_name=self.model_name,
                llm_provider=self.llm_provider,
                temperature=self.llm_temperature,
                max_tokens=self.max_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
                system_prompt=self.system_prompt
           )
           new_instance.chat_model = new_chat_model
           new_instance.chat_prompt_template = self.chat_prompt_template
           return new_instance
        
        else:
            new_chat_model = self.chat_model.bind_tools(tool_list)
            new_instance = CustomChatModel(
                model_name=self.model_name,
                llm_provider=self.llm_provider,
                temperature=self.llm_temperature,
                max_tokens=self.max_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
                system_prompt=self.system_prompt
           )
            new_instance.chat_model = new_chat_model
            new_instance.chat_prompt_template = self.chat_prompt_template
            return new_instance






if __name__ == "__main__":
    load_dotenv()

    # open prompt.txt file
    with open("prompt.txt", "r") as file:
        prompt = file.read()

    chatbot = load_chat_model(
        model_name="meta-llama/Meta-Llama-3-8B",
        llm_provider="huggingface",
        temperature=1.0,
    )

    # we load the prompt from a txt file (test.txt)
    query = "What is the capital of France?"
    print("Chat Template:", chatbot.chat_prompt_template)

    chain = chatbot.chat_prompt_template | chatbot.chat_model

    # print("chain:",chain)

    e = chain.invoke({"text": query})

    print(e)

   # print(e.response_metadata["logprobs"]["content"][:5])