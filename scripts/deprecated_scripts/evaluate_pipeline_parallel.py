import argparse
import csv
import logging
import os
import shutil
import threading
import time
import traceback

import numpy as np
import yaml

# load environment variables
from dotenv import load_dotenv
from tqdm import tqdm

from src.evaluation_utils import answer_relevancy_score_v1, load_json

# custom imports
from src.generation_utils import RAG_answer
from src.retrieval_utils import query_database_v2
from src.utils import (
    log_execution_time,
    text_preprocessing,
)

load_dotenv()


def process_query(query, chunk, config, evaluate_generation, results, logger):
    """
    Function to process a single query.
    """
    start_time = time.time()

    try:
        if evaluate_generation:
            answer, docs, sources = RAG_answer(
                query,
                cot_enabled=config["cot_enabled"],
                nb_chunks=config["nb_chunks"],
                nb_rerank=config["nb_rerank"],
                use_reranker=config["use_reranker"],
                reranker_model=config["reranker_model"],
                embedding_model=config["embedding_model"],
                return_chunks=True,
                model_name=config["model_name"],
                hybrid_search=config["hybrid_search"],
                word_filter=config["word_filter"],
                persist_directory=config["persist_directory"],
                use_multi_query=config["use_multi_query"],
                auto_merging=config["auto_merging"],
                auto_merging_threshold=config["auto_merging_threshold"],
                filter_on_length=config["filter_on_length"],
                length_threshold=config["length_threshold"],
                auto_hybrid_search=config["auto_hybrid_search"],
                advanced_hybrid_search=config["advanced_hybrid_search"],
                source_filter=config["source_filter"],
                source_filter_type=config["source_filter_type"],
                enable_source_filter=config["enable_source_filter"],
                alpha=config["alpha"],
            )
        else:
            answers = query_database(
                query=query,
                nb_chunks=config["nb_chunks"],
                nb_rerank=config["nb_rerank"],
                reranker_model=config["reranker_model"],
                embedding_model=config["embedding_model"],
                use_reranker=config["use_reranker"],
                auto_hybrid_search=config["auto_hybrid_search"],
                source_filter=config["source_filter"],
                source_filter_type=config["source_filter_type"],
                enable_source_filter=config["enable_source_filter"],
                word_filter=config["word_filter"],
                hybrid_search=config["hybrid_search"],
                persist_directory=config["persist_directory"],
                use_multi_query=config["use_multi_query"],
                filter_on_length=config["filter_on_length"],
                length_threshold=config["length_threshold"],
                advanced_hybrid_search=config["advanced_hybrid_search"],
                alpha=config["alpha"],
            )

        end_time = time.time()

        doc_found = False
        for i in range(config["top_k"]):
            if evaluate_generation:
                if docs[i] == text_preprocessing(chunk):
                    if i < 1:
                        results["correct_answers_top1"] += 1
                        doc_found = True
                    if i < 3:
                        results["correct_answers_top3"] += 1
                        doc_found = True
                    if i < 5:
                        results["correct_answers_top5"] += 1
                        doc_found = True
                    break
            else:
                if answers[i].page_content == chunk:
                    if i < 1:
                        results["correct_answers_top1"] += 1
                        doc_found = True
                    if i < 3:
                        results["correct_answers_top3"] += 1
                        doc_found = True
                    if i < 5:
                        results["correct_answers_top5"] += 1
                        doc_found = True
                    break

        if not doc_found:
            logger.info(
                f"Predicted chunk does not match target chunk. PREDICTED: {docs if evaluate_generation else answers}, TARGET: {text_preprocessing(chunk) if evaluate_generation else chunk}, QUERY: {query}"
            )

        if evaluate_generation:
            relevance = answer_relevancy_score_v1(query, answer)
            results["relevance_scores"].append(relevance)

        results["latencies"].append(end_time - start_time)

    except Exception as e:
        tb = traceback.format_exc()
        print("AN ERROR OCCURED DURING THE EVALUATION:", e)
        print("Traceback:", tb)


def launch_eval(config, evaluate_generation=False, logger=None):
    """
    Load the dataset, iterate over the examples by: 1/ Making RAG with each query 2/ Counting the time when the answer is correct
    """

    # Load the parameters from the config file
    persist_directory = config["persist_directory"]

    logger.info("Starting evaluation...")

    # Load the dataset
    dataset_name = persist_directory.split("/")[-1] + "_evaluation_dataset.json"
    dataset_path = os.path.join(persist_directory, dataset_name)
    examples_dict = load_json(dataset_path)

    # Initialize results dictionary
    results = {
        "correct_answers_top1": 0,
        "correct_answers_top3": 0,
        "correct_answers_top5": 0,
        "total_answers": 0,
        "faithfulness_scores": [],
        "relevance_scores": [],
        "latencies": [],
    }

    # Iterate over the examples
    threads = []
    for key, value in tqdm(examples_dict.items()):
        chunk = value["chunk"]
        questions = value["questions"]

        try:
            query_list = list(questions.values())
        except Exception as e:
            print("AN ERROR OCCURED DURING THE EVALUATION:", e)
            query_list = [str(questions)]

        for query in query_list:
            query = str(query)
            results["total_answers"] += 1
            t = threading.Thread(
                target=process_query,
                args=(
                    query,
                    chunk,
                    config,
                    evaluate_generation,
                    results,
                    logger,
                ),
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    logger.info("Evaluation completed.")

    # Calculate accuracies
    accuracy_top1 = results["correct_answers_top1"] / results["total_answers"]
    accuracy_top3 = results["correct_answers_top3"] / results["total_answers"]
    accuracy_top5 = results["correct_answers_top5"] / results["total_answers"]

    # Calculate average latency
    average_latency = np.mean(results["latencies"])

    # Print the results
    print(
        f"Number of correct answers: {results['correct_answers_top1']} out of {results['total_answers']} for top 1"
    )
    print(
        f"Number of correct answers: {results['correct_answers_top3']} out of {results['total_answers']} for top 3"
    )
    print(
        f"Number of correct answers: {results['correct_answers_top5']} out of {results['total_answers']} for top 5"
    )
    print(f"Accuracy top 1: {accuracy_top1*100}%")
    print(f"Accuracy top 3: {accuracy_top3*100}%")
    print(f"Accuracy top 5: {accuracy_top5*100}%")

    # Print the average latency
    print(
        f"Average latency: {average_latency} seconds in case of eval mode: {evaluate_generation}"
    )

    if evaluate_generation:
        print(
            f"Average relevance score: {np.mean(results['relevance_scores'])}"
        )

    # Write the results to a CSV file
    results_file = "evaluation_results.csv"
    temp_file = "temp.csv"
    header = list(config.keys()) + [
        "accuracy_top1",
        "accuracy_top3",
        "accuracy_top5",
        "average_latency",
    ]
    if evaluate_generation:
        header += ["average_relevance_score"]

    if os.path.isfile(results_file):
        with open(results_file, "r") as f, open(
            temp_file, "w", newline=""
        ) as temp:
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
            accuracy_top1,
            accuracy_top3,
            accuracy_top5,
            average_latency,
        ]
        if evaluate_generation:
            row += [np.mean(results["relevance_scores"])]
        writer.writerow(row)

    # We return the metrics as a dictionary
    return {
        "accuracy_top1": accuracy_top1,
        "accuracy_top3": accuracy_top3,
        "accuracy_top5": accuracy_top5,
        "average_latency": average_latency,
    }


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

    parser = argparse.ArgumentParser(
        description="Launch evaluation with a given configuration file."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration file."
    )

    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.safe_load(f)
        print("Config file:", config)

    launch_eval(config, logger=logger)
