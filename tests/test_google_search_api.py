from json import load
from googleapiclient.discovery import build

def google_programmable_search(query, api_key, cse_id, num_results=10):
    """
    Effectue une recherche avec l'API Google Programmable Search.

    Args:
        query (str): La requête de recherche.
        api_key (str): Votre clé d'API Google Cloud.
        cse_id (str): L'identifiant de votre moteur de recherche programmable.
        num_results (int): Le nombre de résultats souhaités.

    Returns:
        list: Une liste d'URL de résultats de recherche.
    """
    service = build("customsearch", "v1", developerKey=api_key)
    results = (
        service.cse()
        .list(q=query, cx=cse_id, num=num_results)
        .execute()
    )

    urls = []
    if "items" in results:
        for item in results["items"]:
            urls.append(item["link"])
    return urls


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    # Remplacez par vos informations réelles
    API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")  # Obtenue depuis Google Cloud Console
    CSE_ID = os.getenv("SEARCH_ENGINE_ID")  # Obtenu depuis le moteur de recherche programmable

    query = "Offre Data Scientist Paris Hello Work"  # La requête de recherche
    num_results = 10  # Le nombre de résultats souhaités

    urls = google_programmable_search(query, API_KEY, CSE_ID, num_results)

    if urls:
        print(f"URLs trouvées pour '{query}':")
        for url in urls:
            print(url)
    else:
        print(f"Aucun résultat trouvé pour '{query}'.")