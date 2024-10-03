import pandas as pd

# Lire le fichier CSV
df = pd.read_csv('evaluation_results.csv')

# Filtrer les lignes selon les critères spécifiés
filtered_df = df[
    (df['clone_embedding_model'].isin(['all-mini-lm-v6', 'bge-M3'])) &
    (df['enable_routing'] == True) &
    (df['hybrid_search'] == True) &
    (df['use_reranker'] == True)
]

# Sélectionner les colonnes pertinentes pour la comparaison
comparison_columns = [
    'clone_embedding_model', 'accuracy_top1', 'accuracy_top3', 'accuracy_top5', 
    'average_latency', 'entity_recall'
]

# Créer un tableau comparatif
comparison_table = filtered_df[comparison_columns]

# Afficher le tableau comparatif
print(comparison_table)