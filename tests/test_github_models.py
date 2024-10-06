import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GITHUB_TOKEN")
endpoint =  "https://models.inference.ai.azure.com"
model_name = "cohere-command-r-plus"   # "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

#measure the time it takes to generate a response
import time
start_time = time.time()
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)
end_time = time.time()


print(response.choices[0].message.content)
print("Time taken to generate response:", end_time-start_time)