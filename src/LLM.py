import os
from functools import lru_cache

import ollama
import torch
import transformers
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_cerebras import ChatCerebras
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# load environment variables
load_dotenv()


def HuggingFaceAnwer(prompt, model_name, temperature=1.0): #PROBABLY DEPRECATED
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=512,  # 256 avant
        eos_token_id=terminators,
        do_sample=True,
        temperature=1,
        top_p=0.9,
    )
    answer = outputs[0]["generated_text"][-1]
    return answer


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

        if llm_provider == "huggingface_api":
            # Initialize HuggingFaceEndpoint
            self.llm = HuggingFaceEndpoint(
                repo_id=f"HuggingFaceH4/{self.llm_name}",
                task="text-generation",
                max_new_tokens=1000,
                top_k=30,
                temperature=self.llm_temperature,
                repetition_penalty=1.03,
            )
            self.context_window_size = 4096

            system = "You are a helpful assistant."  # Define the system message
            human = "{text}"  # Define the human input
            prompt_template = ChatPromptTemplate.from_messages(
                [("system", system), ("human", human)]
            )  # Define the chat prompt template
            self.chat_prompt_template = prompt_template
            self.chat_model = self.llm

        elif self.llm_provider == "groq":
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

        elif self.llm_provider == "langchain":
            # Get the template for the model
            template = ollama.show(self.llm_name)["template"]
            self.llm = ChatOllama(
                model=self.llm_name,
                keep_alive=0,
                num_ctx=8192,  # Example context window size
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
        
        elif self.llm_provider == "cerebras":
          
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
