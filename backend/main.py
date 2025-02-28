from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
from pathlib import Path
import sys
import os
import traceback
import asyncio
import concurrent.futures

print("========= DÉMARRAGE DU SERVEUR BACKEND =========")

# Ajouter le répertoire parent au chemin de recherche des modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(f"Répertoire parent ajouté au chemin: {parent_dir}")

# Import de la fonction LLM_answer_v3 du module de génération
try:
    print("Tentative d'importation de LLM_answer_v3...")
    from src.main_utils.generation_utils_v2 import LLM_answer_v3
    print("Import LLM_answer_v3 réussi!")
except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer LLM_answer_v3: {str(e)}")
    print(traceback.format_exc())
    raise

app = FastAPI()

# Autoriser les requêtes CORS pour le développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger la configuration depuis config.yaml
print("Chargement de la configuration depuis config.yaml...")
config_path = Path(parent_dir) / "config" / "config.yaml"
print(f"Chemin de configuration: {config_path}")

try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Configuration chargée: modèle={config.get('model_name', 'non défini')}, provider={config.get('llm_provider', 'non défini')}")
except Exception as e:
    print(f"ERREUR CRITIQUE: Impossible de charger la configuration: {str(e)}")
    print(traceback.format_exc())
    config = {
        "model_name": "gemini-2.0-flash",
        "llm_provider": "google",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 45
    }
    print(f"Utilisation de la configuration par défaut: {config}")

class Message(BaseModel):
    message: str

# Fonction pour générer une réponse avec un temps maximum
async def generate_response_with_timeout(message_text, max_time=15.0):
    """Génère une réponse avec un timeout maximum"""
    print(f"Génération d'une réponse pour: '{message_text}' avec timeout {max_time} sec")
    
    # Créer une tâche pour la génération de réponse
    loop = asyncio.get_event_loop()
    
    # Définir la fonction qui sera exécutée dans un thread séparé
    def run_llm():
        model_name = config.get("model_name", "gemini-2.0-flash")
        llm_provider = config.get("llm_provider", "google")
        temperature = config.get("temperature", 1.0)
        top_p = config.get("top_p", 0.95)
        top_k = config.get("top_k", 45)
        
        print(f"Paramètres: model={model_name}, provider={llm_provider}, temp={temperature}, top_p={top_p}, top_k={top_k}")
        print("\n>>> DÉBUT APPEL LLM_answer_v3 <<<")
        
        # S'assurer que stream est explicitement mis à False
        stream = False
        print(f"Mode stream: {stream}")
        
        try:
            # Ajouter un tool_list vide pour s'assurer que stream est désactivé
            response = LLM_answer_v3(
                prompt=message_text,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=False,  # Forcer stream=False
                llm_provider=llm_provider,
                tool_list=[],  # Ajouter cette ligne pour désactiver le streaming (selon la logique du code dans generation_utils_v2.py)
            )
            
            print(">>> FIN APPEL LLM_answer_v3 <<<\n")
            print(f"Type de réponse: {type(response)}")
            
            # Vérifier explicitement le type de réponse
            if hasattr(response, '__iter__') and not isinstance(response, (str, bytes, dict)):
                # C'est un générateur ou un itérable, nous devons récupérer tous les morceaux
                print("La réponse est un générateur ou un itérable, conversion en chaîne...")
                full_response = ""
                try:
                    for chunk in response:
                        print(f"Chunk reçu: {chunk}")
                        if isinstance(chunk, dict) and 'chunk' in chunk:
                            full_response += chunk['chunk']
                        else:
                            full_response += str(chunk)
                except Exception as chunk_error:
                    print(f"Erreur lors de la conversion des chunks: {str(chunk_error)}")
                print(f"Réponse complète après conversion: {full_response[:100]}...")
                return full_response
            elif isinstance(response, tuple) and len(response) >= 1:
                # Si c'est un tuple (possible avec tool_calls), prendre le premier élément
                print("La réponse est un tuple, extraction du premier élément...")
                return str(response[0])
            else:
                # C'est déjà une chaîne ou un autre type de données
                print(f"Réponse directe: {str(response)[:100]}...")
                return str(response)
            
        except Exception as e:
            print(f"ERREUR dans LLM_answer_v3: {str(e)}")
            print(traceback.format_exc())
            return f"Erreur lors de la génération de la réponse: {str(e)}"
    
    try:
        # Exécuter la fonction LLM dans un thread séparé avec un timeout
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await asyncio.wait_for(
                loop.run_in_executor(pool, run_llm),
                timeout=max_time
            )
        return result
    except asyncio.TimeoutError:
        print(f"TIMEOUT: La génération a dépassé {max_time} secondes")
        return f"Désolé, la génération de la réponse prend trop de temps. Veuillez essayer une question plus simple ou réessayer plus tard."
    except Exception as e:
        print(f"ERREUR pendant la génération: {str(e)}")
        print(traceback.format_exc())
        return f"Une erreur s'est produite: {str(e)}. Veuillez réessayer."

@app.post("/chat")
async def chat_endpoint(message: Message):
    print("\n" + "="*50)
    print(f"NOUVELLE REQUÊTE: {message.message}")
    print("="*50 + "\n")
    
    try:
        # Générer la réponse avec un timeout de 15 secondes (augmenté pour donner plus de temps au modèle)
        response = await generate_response_with_timeout(message.message, max_time=15.0)
        
        # Vérifier si la réponse est None ou vide
        if response is None:
            print("Réponse vide reçue de LLM_answer_v3")
            return {"response": "Désolé, je n'ai pas pu générer une réponse. Veuillez réessayer."}
        
        # S'assurer que la réponse est bien une chaîne
        response_str = str(response) if response is not None else ""
        print(f"Réponse finale renvoyée: {response_str[:100]}...")
        
        # Vérifier si la réponse contient le format SSE "data: {"chunk":...
        if response_str.startswith("data:"):
            print("ATTENTION: La réponse semble être au format SSE. Correction...")
            # Essayer de nettoyer la chaîne SSE
            try:
                import re
                import json
                # Extraire le contenu entre les accolades
                chunks = re.findall(r'data: ({.*?})', response_str)
                full_text = ""
                for chunk in chunks:
                    try:
                        chunk_data = json.loads(chunk)
                        if "chunk" in chunk_data:
                            full_text += chunk_data["chunk"]
                    except:
                        pass
                
                if full_text:
                    print(f"Réponse convertie du format SSE: {full_text[:100]}...")
                    response_str = full_text
                else:
                    print("Impossible de convertir la réponse SSE")
            except Exception as e:
                print(f"Erreur lors de la conversion SSE: {str(e)}")
        
        return {
            "response": response_str
        }
    except Exception as e:
        print(f"ERREUR GÉNÉRALE: {str(e)}")
        print(traceback.format_exc())
        return {
            "response": f"Erreur: {str(e)}. Veuillez réessayer avec une autre question."
        }

# Vérification de la santé de l'API
@app.get("/health")
async def health_check():
    return {"status": "ok", "config": {"model": config.get("model_name"), "provider": config.get("llm_provider")}}

if __name__ == "__main__":
    print("Ce fichier est exécuté directement, pas via uvicorn.")
    # Ce bloc ne sera pas exécuté lors du démarrage avec uvicorn
