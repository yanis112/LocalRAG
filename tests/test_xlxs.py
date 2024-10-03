import pandas as pd

# Data for the benchmark spreadsheet
data = {
    'Tool': [
        'Google Vertex AI Search',
        'eyelevel.ai',
        'Vectara.com',
        'Own Dev (Internship Project)',
        'Other'
    ],
    'Flexibility of Customization (LLM, Vector-Store, Re-Ranking)': ['', '', '', '', ''],
    'Ease of Integration (Source: GDrive, etc.)': ['', '', '', '', ''],
    'Ease of Chat Integration (Website, Tools)': ['', '', '', '', ''],
    'Access Management (Source Permissions to VectorStore)': ['', '', '', '', ''],
    'Data Privacy with Provider': ['', '', '', '', ''],
    'Price': ['', '', '', '', '']
}

# Create a DataFrame
df_benchmark = pd.DataFrame(data)

# Saving to Excel
file_path_benchmark = 'RAG_as_a_Service_Benchmark.xlsx'
df_benchmark.to_excel(file_path_benchmark, index=False)

file_path_benchmark
