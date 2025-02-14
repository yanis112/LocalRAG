from langchain_google_genai import ChatGoogleGenerativeAI
from src.main_utils.generation_utils_v2 import LLM_answer_v3

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=1
)

answer=LLM_answer_v3(prompt="Combien fait l'intégrale de x^2 dx ?", model_name="gemini-2.0-flash-thinking-exp-01-21",llm_provider="google",stream=True)

print("Answer:",answer)
#développe les tokens du générateur un par un
for k in answer:
    print(k)


# print(llm.invoke(["Combien fait l'intégrale de x^2 dx ?"]))