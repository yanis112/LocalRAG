import transformers

import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

#load prompt from prompt.txt file
with open("prompt.txt", "r") as file:
    prompt = file.read()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=512, #256 avant 
    eos_token_id=terminators,
    do_sample=True,
    temperature=1,
    top_p=0.9,
)
print(outputs[0]["generated_text"][-1])