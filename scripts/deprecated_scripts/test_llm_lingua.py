from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(model_name="openai-community/gpt2")


#load prompt2.txt file

with open("prompt2.txt", "r") as file:
     context = file.read()
    
#context="bonjour, je m'appel yanis et j'habite à marseille"
    

compressed_prompt = llm_lingua.compress_prompt(
    context=context,
    question="Who is he",
    condition_in_question="after_condition",
    dynamic_context_compression_ratio=0.8,
    reorder_context="sort",
    condition_compare=True,
    context_budget="+100",
    rank_method="longllmlingua",
)

print("Compressed Prompt:",compressed_prompt)

print("Nombre de mots dans le prompt original:", len(context.split()))
print("Nombre de mots dans le prompt compressé:", len(compressed_prompt["compressed_prompt"].split()))

