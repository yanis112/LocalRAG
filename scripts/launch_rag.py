import argparse

import yaml
from dotenv import load_dotenv

from src.generation_utils import RAG_answer


def run_rag_pipeline(question, config_file_path):
    # Load the environment variables (API keys,ect...)
    load_dotenv()
    # Open the configuration file and load the different arguments
    with open(config_file_path) as f:
        config = yaml.safe_load(f)

    # Exécutez la fonction RAG_answer avec la question fournie et les arguments du fichier de configuration
    response = RAG_answer(query=question, default_config=config, config={"return_chunks": False})

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exécute le pipeline RAG avec une question et un fichier de configuration."
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="La question à poser au pipeline RAG",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Le fichier de configuration pour le pipeline RAG",
    )

    args = parser.parse_args()

    answer = run_rag_pipeline(args.question, args.config_file)

    print(answer)
