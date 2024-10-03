import argparse
import os
import warnings

import yaml

# custom imports
from src.retrieval_utils import directory_to_vectorstore

warnings.filterwarnings("ignore")


def fill_vectorstore():
    """
    Fills the Chroma database by calling the directory_to_chroma_v2 function with the provided configuration parameters.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Initialise la base de données Chroma."
    )
    parser.add_argument(
        "config_file", type=str, nargs='?', default='config/config.yaml', help="Le nom du fichier de configuration."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Le fichier de configuration {args.config_file} n'existe pas.")
        return

    # Charger les paramètres de configuration
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Appeler la fonction directory_to_chroma_v2 avec les paramètres de configuration
    directory_to_vectorstore(default_config=config, config={})


if __name__ == "__main__":
    fill_vectorstore()
