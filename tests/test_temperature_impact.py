import numpy as np
import matplotlib.pyplot as plt
import ollama
from tqdm import tqdm
from main_utils.generation_utils import LLM_answer,LLM_answer_v3

# Function to generate output and count tokens
def generate_and_count_tokens(prompt, temperature):
    response = LLM_answer_v3(
        prompt=prompt,
        model_name='llama3.1',
        temperature=temperature,
        stream=False,
        llm_provider='ollama'
    )
    return len(response.split())  # Count the number of tokens

# Initialize variables
temperatures = np.linspace(0, 3, 10)  # 10 temperature values from 0 to 3
token_counts = []

# Conduct 10 runs for each temperature
for temp in tqdm(temperatures,desc='Processing...'):
    temp_runs = []
    for _ in range(20):
        token_count = generate_and_count_tokens('Why is the sky blue?', temp)
        temp_runs.append(token_count)
    mean_tokens = np.mean(temp_runs)
    token_counts.append(mean_tokens)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(temperatures, token_counts, marker='o')
plt.title('Mean Number of Output Tokens vs. LLM Temperature')
plt.xlabel('LLM Temperature')
plt.ylabel('Mean Number of Output Tokens')
plt.xlim(0, 3)
plt.ylim(min(token_counts)*0.9, max(token_counts)*1.1)
plt.grid(True)
plt.axhline(y=np.mean(token_counts), color='r', linestyle='--', label='Mean Output Token Count')
plt.legend()
plt.savefig('output_token_count_vs_temperature.png')
plt.show()