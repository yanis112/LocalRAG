import threading
import time
from src.generation_utils import LLM_answer

# Fonction pour traiter une requête
def process_request(prompt):
    response = LLM_answer(
        prompt=prompt,
        model_name='llama3.1',
        temperature=1,
        stream=False,
        llm_provider='ollama'
    )
    return response

# Fonction pour exécuter des requêtes en parallèle avec threading
def process_requests_parallel(prompts):
    results = [None] * len(prompts)
    threads = []

    def worker(i, prompt):
        results[i] = process_request(prompt)

    for i, prompt in enumerate(prompts):
        thread = threading.Thread(target=worker, args=(i, prompt))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results

# Fonction pour traiter une requête de manière séquentielle
def process_requests_sequential(prompts):
    results = []
    for prompt in prompts:
        results.append(process_request(prompt))
    return results

# Liste des prompts à traiter
prompts = [
    "Pourquoi le ciel est-il bleu pendant la journée et rouge au coucher du soleil?",
    "Quels sont les impacts du changement climatique sur les écosystèmes marins?",
    "Comment fonctionne l'intelligence artificielle et quelles sont ses applications dans la vie quotidienne?"
]

# Nombre de runs pour la moyenne
num_runs = 10

# Mesure du temps pour les requêtes parallèles
total_time_parallel = 0
for _ in range(num_runs):
    start_parallel = time.time()
    results_parallel = process_requests_parallel(prompts)
    end_parallel = time.time()
    total_time_parallel += end_parallel - start_parallel

average_time_parallel = total_time_parallel / num_runs

# Affichage des résultats parallèles
print("Parallel Execution Results:")
for prompt, result in zip(prompts, results_parallel):
    print(f"Prompt: {prompt}")
    print(f"Response: {result}")
    print("-" * 40)
print(f"Average total time for parallel execution: {average_time_parallel:.2f} seconds")
print("=" * 40)

# Mesure du temps pour les requêtes séquentielles
total_time_sequential = 0
for _ in range(num_runs):
    start_sequential = time.time()
    results_sequential = process_requests_sequential(prompts)
    end_sequential = time.time()
    total_time_sequential += end_sequential - start_sequential

average_time_sequential = total_time_sequential / num_runs

# Affichage des résultats séquentiels
print("Sequential Execution Results:")
for prompt, result in zip(prompts, results_sequential):
    print(f"Prompt: {prompt}")
    print(f"Response: {result}")
    print("-" * 40)
print(f"Average total time for sequential execution: {average_time_sequential:.2f} seconds")
print("=" * 40)