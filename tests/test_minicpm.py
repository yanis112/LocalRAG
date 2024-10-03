import ollama

with open('assets/retrieval_comparison_results_100ex.png', 'rb') as image_file:
    image_data = image_file.read()
    response = ollama.chat(model='minicpm-v', messages=[
        {
            'role': 'user',
            'content': 'Quel est la méthode la plus efficace d après ce graphique:',
            'images': [image_data]
        }
    ])

print(response['message']['content'])