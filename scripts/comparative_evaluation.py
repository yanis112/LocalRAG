from evaluate_pipeline_v2 import launch_eval
import os
import yaml
import logging
import sys

if __name__ == "__main__":
    # Create a new logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers from the logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a file handler
    handler = logging.FileHandler("evaluation.log")
    handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter("%(asctime)s %(message)s\n")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Get the path from command line arguments or use default
    path = sys.argv[1] if len(sys.argv) > 1 else "config/comparative_evaluation"

    # We find all the yaml files in the specified path
    list_files = os.listdir(path)
    # For each one, we load the config yaml file and launch the evaluation
    for file in list_files:
        current_config_path = os.path.join(path, file)
        print(f"Launching evaluation with config file: {current_config_path} ...")
        config = yaml.safe_load(open(current_config_path, "r"))
        launch_eval(config, logger=logger)