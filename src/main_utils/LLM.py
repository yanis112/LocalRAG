import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate


# load environment variables
load_dotenv()


class CustomChatModel:
    """
    Represents a custom chat model.

    Args:
        llm_name (str): The name of the language model.
        llm_provider (str): The provider of the language model.
        temperature (float, optional): The temperature for text generation. Defaults to 1.0.
    """

    @lru_cache(maxsize=0)
    def __init__(self, llm_name, llm_provider, temperature=1.0):
        self.llm_name = llm_name
        self.llm = None
        self.chat_model = None
        self.chat_prompt_template = None
        self.llm_temperature = temperature
        self.llm_provider = llm_provider
        self.system_prompt = "You are a helpful assistant."
        

    

        if self.llm_provider == "groq":
            from langchain_groq import ChatGroq
            self.context_window_size = 131072
            self.chat_model = ChatGroq(
                temperature=self.llm_temperature,
                model_name=self.llm_name,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            )
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{text}"),
                ]
            )

        elif self.llm_provider == "ollama":
            import ollama
            from langchain_ollama import ChatOllama
            template = ollama.show(self.llm_name)["template"]
            self.context_window_size = 8192
            self.llm = ChatOllama(
                model=self.llm_name,
                keep_alive=0,
                num_ctx=self.context_window_size,
                temperature=self.llm_temperature,
                template=template,
            )
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{text}"),
                ]
            )
            self.chat_model = self.llm

        elif self.llm_provider == "huggingface":
            # Find all available cuda devices
            import torch
            import transformers
            pipeline = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]

            self.llm = HuggingFacePipeline.from_model_id(
                model_id=self.llm_name,
                task="text-generation",
                device_map="auto",
                pipeline_kwargs=dict(
                    max_new_tokens=1024,
                    temperature=self.llm_temperature,
                    torch_dtype=torch.bfloat16,
                    eos_token_id=terminators,
                    do_sample=True,
                    top_p=0.9,
                ),
            )

            self.chat_model = ChatHuggingFace(llm=self.llm)

            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{text}"),
                ]
            )

       
        elif self.llm_provider == "cerebras":
            from langchain_cerebras import ChatCerebras
          
            #use load_dotenv() to load the environment variables
            os.environ["CEREBRAS_API_KEY"] = os.getenv("CEREBRAS_API_KEY")

            self.llm = ChatCerebras(
                model=self.llm_name,
            )
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant. You are requiered to give any information an user requires."),
                    ("human", "{text}"),
                ]
            )
            self.chat_model = self.llm
            
        elif self.llm_provider == "sambanova":
            from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
            os.environ["SAMBANOVA_API_KEY"] = os.getenv("SAMBANOVA_API_KEY")
            self.context_window_size = 8000  # Example context window size
            self.chat_model = ChatSambaNovaCloud(
                model=self.llm_name,
                max_tokens=1024,
                temperature=self.llm_temperature,
                top_k=1,
                top_p=0.01,
            )
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{text}"),
                ]
            )
            
        elif self.llm_provider == "github":
            from src.aux_utils.github_llm import GithubLLM
            self.llm = GithubLLM(
                github_token=os.getenv("GITHUB_TOKEN"),
                model_name=self.llm_name,
                temperature=self.llm_temperature,
                top_p=1.0,
                max_tokens=1000,
            )
            
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{text}"),
                ]
            )
            
        elif self.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=self.llm_name,
                temperature=self.llm_temperature,
                max_tokens=3000,
                timeout=None,
                max_retries=1,
                # other params...
            )
            
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("human", "{text}"),
                ]
            )
            self.chat_model = self.llm
            print("CHAT MODEL:", self.chat_model)
            

        else:
            raise ValueError(
                "CRITICAL ERROR: THE LLM MODEL NAME IS NOT RECOGNIZED. VERIFY THE LLM NAME (OR LLM PROVIDER NAME) !"
            )


if __name__ == "__main__":
    load_dotenv()

    # open prompt.txt file
    with open("prompt.txt", "r") as file:
        prompt = file.read()

    chatbot = CustomChatModel(
        llm_name="meta-llama/Meta-Llama-3-8B",
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

    print(e.response_metadata["logprobs"]["content"][:5])
