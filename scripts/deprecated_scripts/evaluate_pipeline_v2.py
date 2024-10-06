import argparse
import csv
import logging
import os
import shutil
import time
import traceback

import numpy as np
import yaml
import mlflow

from dotenv import load_dotenv
import json 
import pandas as pd
from tqdm import tqdm

from src.retrieval_utils import query_database_v2
from src.evaluation_utils import answer_relevancy_score_v1, load_json, mean_hallucination_score, entity_recall
from src.generation_utils import RAG_answer

load_dotenv()

def load_dataset(config):
    """
    Load a dataset from a JSON file and return it as a pandas DataFrame.

    Args:
        config (dict): Configuration dictionary containing the key "persist_directory",
                       which specifies the directory where the dataset is stored.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame containing the dataset.
            - str: The path to the dataset file.
    """
    persist_directory = config["persist_directory"]
    dataset_path = os.path.join(persist_directory, persist_directory.split('/')[-1] + '_evaluation_dataset.json')
    with open(dataset_path, 'r') as file:
        json_data = json.load(file)
    df = pd.DataFrame.from_dict(json_data, orient='index')
    df['questions'] = df['questions'].astype(str)
    return df, dataset_path

def log_metrics(metrics, config, metrics_list, evaluate_generation):
    """
    Log various metrics to MLflow.

    Parameters:
        metrics (dict): A dictionary containing metric names as keys and their values.
        config (dict): Configuration dictionary (not used in the function but kept for consistency).
        metrics_list (list): A list of metric names to be logged conditionally.
        evaluate_generation (bool): A flag indicating whether to log generation-specific metrics.

    Metrics Logged:
        - accuracy_top1
        - accuracy_top3
        - accuracy_top5
        - entity_recall
        - average_latency (if 'latency' is in metrics_list)
        - average_relevance_score (if evaluate_generation is True)
        - average_hallucination_score (if evaluate_generation is True)
    """
    mlflow.log_metric("accuracy_top1", metrics["accuracy_top1"])
    mlflow.log_metric("accuracy_top3", metrics["accuracy_top3"])
    mlflow.log_metric("accuracy_top5", metrics["accuracy_top5"])
    mlflow.log_metric("entity_recall", metrics["entity_recall"])
    if 'latency' in metrics_list:
        mlflow.log_metric("average_latency", metrics["average_latency"])
    if evaluate_generation:
        mlflow.log_metric("average_relevance_score", metrics["average_relevance_score"])
        mlflow.log_metric("average_hallucination_score", metrics["average_hallucination_score"])

def write_results_to_csv(metrics, config, metrics_list, evaluate_generation):
    """
    Write evaluation metrics to a CSV file.

    This function writes the evaluation metrics to a CSV file named 
    "evaluation_results.csv". If the file already exists and is not empty, 
    it appends the new results to the file. If the file does not exist or 
    is empty, it creates a new file with the appropriate header.

    Args:
        metrics (dict): A dictionary containing the evaluation metrics.
            Expected keys are "accuracy_top1", "accuracy_top3", "accuracy_top5", 
            "entity_recall", and optionally "average_latency", "average_relevance_score", 
            and "average_hallucination_score".
        config (dict): A dictionary containing the configuration parameters.
            The keys of this dictionary will be used as part of the CSV header.
        metrics_list (list): A list of metric names to be included in the CSV.
            If "latency" is in this list, "average_latency" will be included in the CSV.
        evaluate_generation (bool): A flag indicating whether generation metrics 
            ("average_relevance_score" and "average_hallucination_score") should be included.

    Raises:
        IOError: If there is an issue reading or writing the CSV file.
    """
    results_file = "evaluation_results.csv"
    temp_file = "temp.csv"
    header = list(config.keys()) + [
        "accuracy_top1",
        "accuracy_top3",
        "accuracy_top5",
        "entity_recall"
    ]
    if "latency" in metrics_list:
        header += ["average_latency"]
    if evaluate_generation:
        header += ["average_relevance_score", "average_hallucination_score"]

    if os.path.isfile(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, "r") as f, open(temp_file, "w", newline="") as temp:
            reader = csv.reader(f)
            writer = csv.writer(temp)
            first_line = next(reader)
            if first_line != header:
                writer.writerow(header)
            writer.writerow(first_line)
            for line in reader:
                writer.writerow(line)
        shutil.move(temp_file, results_file)
    else:
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        row = list(config.values()) + [
            metrics["accuracy_top1"],
            metrics["accuracy_top3"],
            metrics["accuracy_top5"],
            metrics["entity_recall"]
        ]
        if "latency" in metrics_list:
            row += [metrics["average_latency"]]
        if evaluate_generation:
            row += [metrics["average_relevance_score"], metrics["average_hallucination_score"]]
        writer.writerow(row)

def evaluate_example(query, chunk, config, top_k, metrics_list, evaluate_generation):
    correct_answers_top1 = 0
    correct_answers_top3 = 0
    correct_answers_top5 = 0
    hallucination_scores = []
    relevance_scores = []
    latencies = []
    entity_recalls = []
    total_answers = 1  # Increment total_answers for each query evaluated

    try:
        start_time = time.time()
        if evaluate_generation:
            answer, docs, sources = RAG_answer(query, default_config=config, config={"return_chunks": True})
        else:
            answers = query_database_v2(query, default_config=config, config={})
            docs = [i.page_content for i in answers]
        end_time = time.time()

        doc_found = False
        for i in range(top_k):
            if docs[i] == chunk:
                doc_found = True
                if i < 1:
                    correct_answers_top1 += 1
                if i < 3:
                    correct_answers_top3 += 1
                if i < 5:
                    correct_answers_top5 += 1
                break

        if "answer_relevancy" in metrics_list and evaluate_generation:
            relevance = answer_relevancy_score_v1(query, answer)
            relevance_scores.append(relevance)
        if "latency" in metrics_list:
            latencies.append(end_time - start_time)
        if "hallucination_score" in metrics_list and evaluate_generation:
            hallucination_score = mean_hallucination_score(answer, docs)
            hallucination_scores.append(hallucination_score)
        if "entity_recall" in metrics_list:
            recall = entity_recall(chunk, docs)
            entity_recalls.append(recall)

    except Exception as e:
        tb = traceback.format_exc()
        print("AN ERROR OCCURED DURING THE EVALUATION:", e)
        print("Traceback:", tb)

    return {
        "correct_answers_top1": correct_answers_top1,
        "correct_answers_top3": correct_answers_top3,
        "correct_answers_top5": correct_answers_top5,
        "latencies": latencies,
        "hallucination_scores": hallucination_scores,
        "relevance_scores": relevance_scores,
        "entity_recalls": entity_recalls,
        "total_answers": total_answers
    }

def calculate_metrics(results):
    total_answers = results["total_answers"]
    accuracy_top1 = results["correct_answers_top1"] / total_answers
    accuracy_top3 = results["correct_answers_top3"] / total_answers
    accuracy_top5 = results["correct_answers_top5"] / total_answers
    average_latency = np.mean(results["latencies"])
    average_relevance_score = np.mean(results["relevance_scores"])
    average_hallucination_score = np.mean(results["hallucination_scores"])
    average_entity_recall = np.mean(results["entity_recalls"])

    return {
        "accuracy_top1": accuracy_top1,
        "accuracy_top3": accuracy_top3,
        "accuracy_top5": accuracy_top5,
        "average_latency": average_latency,
        "average_relevance_score": average_relevance_score,
        "average_hallucination_score": average_hallucination_score,
        "entity_recall": average_entity_recall
    }

def launch_eval(config, logger=None):
    df, dataset_path = load_dataset(config)
    dataset = mlflow.data.from_pandas(df, source=dataset_path)

    with mlflow.start_run():
        mlflow.log_input(dataset, context='evaluation')
        for key, value in config.items():
            mlflow.log_param(key, value)

        top_k = config["top_k"]
        evaluate_generation = config["evaluate_generation"]
        metrics_list = config["metrics_list"]
        num_runs = config.get("num_runs", 1)
        logger.info("Starting evaluation...")

        results = {
            "correct_answers_top1": 0,
            "correct_answers_top3": 0,
            "correct_answers_top5": 0,
            "latencies": [],
            "hallucination_scores": [],
            "relevance_scores": [],
            "entity_recalls": [],
            "total_answers": 0
        }

        for _ in range(num_runs):
            for key, value in tqdm(df.iterrows(), desc="Evaluating"):
                chunk = value["chunk"]
                questions = value["questions"]
                query_list = [str(questions)]
                for query in query_list:
                    query = str(query)
                    example_results = evaluate_example(query, chunk, config, top_k, metrics_list, evaluate_generation)
                    for k, v in example_results.items():
                        if isinstance(v, list):
                            results[k].extend(v)
                        else:
                            results[k] += v

        metrics = calculate_metrics(results)
        log_metrics(metrics, config, metrics_list, evaluate_generation)
        write_results_to_csv(metrics, config, metrics_list, evaluate_generation)

        return metrics
    
if __name__ == "__main__":
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.FileHandler("evaluation.log")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s\n")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    parser = argparse.ArgumentParser(
        description="Launch evaluation with a given configuration file."
    )
    
    parser.add_argument(
        "config_file", type=str, nargs='?', default="config/config.yaml", help="Path to the configuration file."
    )

    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    launch_eval(config, logger=logger)