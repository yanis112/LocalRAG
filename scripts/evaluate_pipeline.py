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

# load environment variables
from dotenv import load_dotenv
import json 
import pandas as pd
from tqdm import tqdm

from src.retrieval_utils import query_database_v2
from src.evaluation_utils import answer_relevancy_score_v1, load_json, mean_hallucination_score, entity_recall
 
# custom imports
from src.generation_utils import RAG_answer

load_dotenv()

def launch_eval(config, logger=None):
    """
    Load the dataset, iterate over the examples by: 1/ Making RAG with each query 2/ Counting the time when the answer is correct
    """


    # Load the JSON dataset
    persist_directory = config["persist_directory"]
    dataset_path = str(persist_directory)+'/'+persist_directory.split('/')[-1]+'_evaluation_dataset.json'
    
    with open(dataset_path, 'r') as file:
        json_data = json.load(file)
    
    # Convert JSON data to Pandas DataFrame
    df = pd.DataFrame.from_dict(json_data, orient='index')
    
    # Ensure all values in the 'questions' column are strings
    df['questions'] = df['questions'].astype(str)
    
    # Convert the DataFrame to a MLflow dataset
    dataset = mlflow.data.from_pandas(df, source=dataset_path)
    
    # Start an MLflow run
    with mlflow.start_run():
        #add the dataset to the run
        mlflow.log_input(dataset,context='evaluation')
        # Log parameters
        for key, value in config.items():
            mlflow.log_param(key, value)
        
        # Load the parameters from the config file
        persist_directory = config["persist_directory"]
        top_k = config["top_k"]
        evaluate_generation = config["evaluate_generation"]
        metrics_list = config["metrics_list"]
        logger.info("Starting evaluation...")
        
        # Load the dataset
        dataset_name = persist_directory.split("/")[-1] + "_evaluation_dataset.json"
        dataset_path = os.path.join(persist_directory, dataset_name)
        examples_dict = load_json(dataset_path)
        
        # Initialize counters
        correct_answers_top1 = 0
        correct_answers_top3 = 0
        correct_answers_top5 = 0
        total_answers = 0
        
        # Initialize lists for scores
        hallucination_scores = []
        relevance_scores = []
        latencies = []
        entity_recalls=[]
        
        # Iterate over the examples
        for key, value in tqdm(examples_dict.items(), desc="Evaluating"):
            chunk = value["chunk"]
            questions = value["questions"]
            query_list = [str(questions)]
            for query in query_list:
                query = str(query)
                total_answers += 1
                if evaluate_generation:
                    try:
                        start_time = time.time()
                        answer, docs, sources = RAG_answer(
                            query,
                            default_config=config,
                            config={"return_chunks": True},
                        )
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
                        if not doc_found:
                            logger.info(
                                f"Predicted chunk does not match target chunk. PREDICTED: {docs}, TARGET: {chunk}, QUERY: {query}"
                            )
                        if "answer_relevancy" in metrics_list:
                            relevance = answer_relevancy_score_v1(query, answer)
                            relevance_scores.append(relevance)
                        if "latency" in metrics_list:
                            latencies.append(end_time - start_time)
                        if "hallucination_score" in metrics_list:
                            hallucination_score = mean_hallucination_score(answer, docs)
                            hallucination_scores.append(hallucination_score)
                        if "entity_recall" in metrics_list:
                            pass
                      
                            
                          
                            
                            
                    except Exception as e:
                        tb = traceback.format_exc()
                        print("AN ERROR OCCURED DURING THE EVALUATION:", e)
                        print("Traceback:", tb)
                        continue
                else:
                    try:
                        start_time = time.time()
                        answers = query_database_v2(
                            query, default_config=config, config={}
                        )
                        #make a list of str docs topk
                        list_docs=[i.page_content for i in answers]
                        end_time = time.time()
                        doc_found = False
                        for i in range(top_k):
                            if answers[i].page_content == chunk:
                                if i < 1:
                                    correct_answers_top1 += 1
                                    doc_found = True
                                if i < 3:
                                    correct_answers_top3 += 1
                                    doc_found = True
                                if i < 5:
                                    correct_answers_top5 += 1
                                    doc_found = True
                                break
                        if not doc_found:
                            logger.info(
                                f"Predicted chunk does not match target chunk. PREDICTED: {answers}, TARGET: {chunk}, QUERY: {query}"
                            )
                            
                            
                        if "entity_recall" in metrics_list:
                            #print("Chunk:",chunk)
                            
                            recall=entity_recall(chunk,list_docs)
                            print("Recall:",recall)
                            entity_recalls.append(recall)
                        latencies.append(end_time - start_time)
                    except Exception as e:
                        print("AN ERROR OCCURED DURING THE EVALUATION:", e)
                        continue
        logger.info("Evaluation completed.")
        
        # Calculate accuracies
        accuracy_top1 = correct_answers_top1 / total_answers
        accuracy_top3 = correct_answers_top3 / total_answers
        accuracy_top5 = correct_answers_top5 / total_answers
        
        # Calculate average latency
        average_latency = np.mean(latencies)
        
        
        
        # Log metrics
        mlflow.log_metric("accuracy_top1", accuracy_top1)
        mlflow.log_metric("accuracy_top3", accuracy_top3)
        mlflow.log_metric("accuracy_top5", accuracy_top5)
        mlflow.log_metric("entity_recall", np.mean(entity_recalls))
        if 'latency' in metrics_list:
            mlflow.log_metric("average_latency", average_latency)
        if evaluate_generation:
            mlflow.log_metric("average_relevance_score", np.mean(relevance_scores))
            mlflow.log_metric("average_hallucination_score", np.mean(hallucination_scores))
        
        # Print the results
        print(f"Number of correct answers: {correct_answers_top1} out of {total_answers} for top 1")
        print(f"Number of correct answers: {correct_answers_top3} out of {total_answers} for top 3")
        print(f"Number of correct answers: {correct_answers_top5} out of {total_answers} for top 5")
        print(f"Accuracy top 1: {accuracy_top1*100}%")
        print(f"Accuracy top 3: {accuracy_top3*100}%")
        print(f"Accuracy top 5: {accuracy_top5*100}%")
        if 'latency' in metrics_list:
            print(f"Average latency: {average_latency} seconds in case of eval mode: {evaluate_generation}")
        if evaluate_generation:
            print(f"Average relevance score: {np.mean(relevance_scores)}")
            print(f"Average hallucination score: {np.mean(hallucination_scores)}")
        
        # Write the results to a CSV file
        results_file = "evaluation_results.csv"
        temp_file = "temp.csv"
        header = list(config.keys()) + [
            "accuracy_top1",
            "accuracy_top3",
            "accuracy_top5"
        ]
        if "latency" in metrics_list:
            header += ["average_latency"]
        if evaluate_generation:
            header += ["average_relevance_score"]
        if "hallucination_score" in metrics_list:
            header += ["average_hallucination_score"]
        if "entity_recall" in metrics_list:
            header += ["entity_recall"]
            
            
            
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
                accuracy_top1,
                accuracy_top3,
                accuracy_top5
            ]
            if "latency" in metrics_list:
                row += [average_latency]
            if evaluate_generation:
                row += [np.mean(relevance_scores)]
            if "hallucination_score" in metrics_list:
                row += [np.mean(hallucination_scores)]
            if "entity_recall" in metrics_list:
                row += [np.mean(entity_recalls)]
            writer.writerow(row)
        
        # Return the metrics as a dictionary
        metrics = {
            "accuracy_top1": accuracy_top1,
            "accuracy_top3": accuracy_top3,
            "accuracy_top5": accuracy_top5,
        }
        if "latency" in metrics_list:
            metrics["average_latency"] = average_latency
        if evaluate_generation:
            metrics["average_relevance_score"] = np.mean(relevance_scores)
        if "hallucination_score" in metrics_list:
            metrics["average_hallucination_score"] = np.mean(hallucination_scores)
        if "entity_recall" in metrics_list:
            metrics["entity_recall"]=np.mean(entity_recalls)
        
        return metrics

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
        "config_file", type=str, nargs='?', default="config/config.yaml", help="Path to the configuration file."
    )

    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    launch_eval(config, logger=logger)