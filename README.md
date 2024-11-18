# LLMs and RAG

This project is a **Retrieval Augmented Generation (RAG)** based question-answering system that uses the Qdrant database to answer questions based on the content of all Euranova's documents and available information. The system utilizes a Language Model (LLM) to generate answers to questions and a retrieval pipeline to fetch relevant information from the database. It can answer simple questions based on the content of the documents and can also handle complex questions by breaking them down into sub-questions and generating answers for each.

<img src="assets/logo_v4.png" alt="Logo" width="300" height="300">

## 🔑 Key Features

- **Simple Question Answering**: Uses the HuggingFace language model to answer questions based on the content of documents loaded into the Chroma database.
- **Advanced Question Answering**: Can answer complex questions by decomposing them into sub-questions and aggregating answers from multiple sources.
- **Evaluation**: Provides scripts for evaluating the performance of the language model.
- **Document Loading**: Functions for loading documents into the Chroma database.
- **Streamlit Interface**: Offers a Streamlit interface for user interaction with the application.
- **Audio Transcription**: Transcribes audio files to text using an optimized implementation of the WhisperV3 model and PyAnnote speaker diarization model.
- **Optical Character Recognition (OCR)**: Extracts text from images using a state-of-the-art OCR model, extracts information from charts and tables, and generates summaries.

## 🛠️ Installation

# Activate venv
 .\.venv\Scripts\Activate.ps1

1. **Clone the Repository**

   ```bash
   git clone [repository_url]
   ```

2. **Navigate to the Project Root Directory**

   ```bash
   cd llms-rag
   $env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"
   ```

3. **Build the Docker Image**

   ```bash
   docker build -t llms-rag/demo:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
   ```

   *Optional: Build from scratch (not usually needed)*

   ```bash
   docker build --no-cache -t llms-rag/demo:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
   ```

4. **Run the Docker Container**

   *Specify the device you want to use with the `--gpus` flag:*

   ```bash
   docker run -d --rm --gpus '"device=0"' -v $(pwd):/home/user/llms --name=llms llms-rag/demo:latest 
   ```

   *For a permanent run:*

   ```bash
   docker run -d --restart=always --gpus '"device=0"' -v $(pwd):/home/user/llms -p 8501:8501 --name=llms llms-rag/demo:latest  
   ```

  ```bash
   docker run -d --restart=always -v $(pwd):/home/user/llms -p 8501:8501 --name=llms llms-rag/demo:latest  
   ```

5. **Run the Streamlit Application**

   ```bash
   streamlit run streamlit_app.py --browser.serverAddress 0.0.0.0
   ```

   *If streamlit asks you to enter your email just press enter to pass this step.*

   *From this container, follow the instructions below to run the project.*

## 📦 Setup

First, you need to fill the `env` file with the following variables:

```bash
GROQ_API_KEY="..."
LANGCHAIN_API_KEY="..."
LANGCHAIN_END_POINT="https://api.smith.langchain.com"
LANGCHAIN_TRACING_v2="true"
HUGGINGFACEHUB_API_TOKEN="..."
GITLAB_URL='https://code.euranova.eu/'  
PRIVATE_TOKEN='...' 
STREAMLIT_URL='http://10.0.13.231:8501'
ALLOWED_FORMATS=["pdf", "docx", "xlsx", "html", "pptx", "txt", "md"]
URL_MATTERMOST="chat.euranova.eu"
MATTERMOST_TOKEN="7u5pdaay5jrbmppqxiubwamklp4h"
```

- **GROQ_API_KEY**: Obtain from the Groq Cloud website.
- **LANGCHAIN_API_KEY**: Obtain from the LangChain website.
- **HUGGINGFACEHUB_API_TOKEN**: Obtain from the HuggingFace website (allow all permissions).
- **GITLAB_URL** and **PRIVATE_TOKEN**: Used to access the Euranova GitLab repository for gitlab, you need to go to your gitlab settings and create a new personal access token with the `api` scope, and paste it here.
- **STREAMLIT_URL**: URL of the Streamlit application.
- **ALLOWED_FORMATS**: Specifies the allowed file formats for documents (no other formats are supported).
- **MATTERMOST_TOKEN**: Access token for Mattermost, you can obtain it by going to a mattermost channel (e.g., `https://chat.euranova.eu/euranova/channels/town-square`) and left click on 'inspect element' and go to the 'network' tab, click on any place on the page, click on any connection you see in the 'network' tab, and look for the cookies section and select the MMAUTHTOKEN	cookie, copy the value and paste it here.

Once you have filled the `env` file, rename it to `.env`.

## 🚀 Quickstart

Here's a summary of the most important steps to start using the project:

1. **Ensure the Docker Container is Running**

   Make sure the Docker container is up and running.

2. **Verify the Project Root**

   Ensure you are at the root of the project.

3. **Prepare the Vector Store**

   - Verify that a vector store containing your data exists in the `data/vector_stores/` directory and that its name is specified in the configuration file under the `persist_directory` parameter.
   - If it does not exist, you will need to create it (requires GPU!). To create it, ensure you have a `data` directory containing all the data you want to include in the index at the root of the project. Ensure the path to this directory is specified in the configuration file under the `path` parameter.
   - Then run the following command to fill the database:

     ```bash
     python scripts/fill_database.py
     ```

4. **Run the Streamlit Application**

   Once the previous steps are completed, run the Streamlit application:

   ```bash
   streamlit run scripts/streamlit_app.py
   ```

5. **Interact with the RAG Chatbot**

   Open the Streamlit application using the local link and use the interface to interact with the RAG chatbot.

**Note:** You may experience delays when running the application for the first time as the models are being loaded.

## 🖱️ How to Use the Streamlit Interface

1. **Launch the Streamlit Application**

   Run the application using:

   ```bash
   streamlit run streamlit_app.py
   ```

2. **Use the Interface**

   - Use the `Advanced search` toggle to enable or disable advanced search features.
   - Type your question in the text box.
   - Click the `Ask` button to get an answer.

## 📁 Directory Structure

The project is organized as follows:

- **Root Directory**

  - `README.md`: The main documentation file for the project.
  - `data/vector_stores/`: Directory containing the different vector stores for the documents, their log files, and respective evaluation datasets. The name of the vector store includes the vector store provider, type of chunking, and the embedding model used.
  - `config/config.yaml`: Configuration file for the project. All parameters for evaluation and retrieval are stored here.
  - `.env`: File containing the environment variables for the project (API keys, etc.).

- **`src` Directory**

  - `retrieval_utils.py`: Functions for loading documents into the Chroma database/filling the database.
  - `data/`: Directory containing all the data: drive_docs, HTML pages, vector_stores, drive_happeo.
  - `evaluate_pipeline.py`: Functions for evaluating the performance of the language model.
  - `generation_utils.py`: Implementation of the question-answering model/RAG pipeline and associated functions.
  - `utils.py`: Various utility functions used throughout the project.
  - `knowledge_graph.py`: Functions to create a knowledge graph from the documents.
  - `LLM.py`: Implementation of the language model at the base of the project and the way to access the different LLMs implemented (through Ollama or LangChain/Groq API).
  - `data_archive.py`: Functions to process special JSON data from Euranova's archives.

- **`scripts` Directory**

  - `launch_rag.py`: CLI command to launch the RAG model.
  - `fill_database.py`: CLI command to fill the Chroma database/initialize it from a directory.
  - `streamlit_app.py`: Contains the Streamlit interface for the RAG chatbot and launches it.
  - `evaluation_results.csv`: CSV file containing the results of the evaluation with different settings/hyperparameters.
  - `happeo_scrapping.py`: Implementation of functions to scrape Happeo pages.
  - `mattermost_scrapping.py`: Implementation of functions to scrape Mattermost channels.
  - `gitlab_scrapping.py`: Implementation of functions to scrape GitLab repositories' READMEs.
  - `comparative_evaluation.py`: Script to perform comparative evaluation of different methods based on a set of predefined config files.
  - `evaluate_pipeline.py`: Functions for evaluating the performance of the language model and the retrieval pipeline.
  - `clustering_by_topic.py`: Script to cluster document chunks by topic using BERTopic and Qdrant.
  - `eval_dataset_curation_streamlit.py`: Streamlit interface for generating evaluation datasets by sampling random chunks from the Chroma or Qdrant database and generating artificial questions for each chunk.

## 🔄 Fill the Qdrant Database

The `fill_database` command is used to initialize the Qdrant database. It takes as an argument the name of the YAML configuration file that contains the necessary parameters for creating the database if it does not already exist and fills it with the documents from the specified directory. Documents already in the database will not be added again.

Here's how you can use this command:

```bash
python scripts/fill_database.py config.yaml
```

Or, simply:

```bash
python scripts/fill_database.py
```

**Note:** Replace `config/config.yaml` with the path to your configuration file.

### Command Details

The `fill_database` command performs the following operations:

1. Reads the YAML configuration file specified as an argument.
2. Checks if the configuration file exists; if not, it displays an error message and terminates.
3. Loads the configuration parameters from the YAML file.
4. Calls the `directory_to_vectorstore` function with the loaded configuration parameters to create the Chroma/Qdrant database using the specified parameters.

### 🔧 Configuration Parameters

Here are the configuration parameters that you can define in the YAML file:

- **Database Initialization Parameters**
  - `config_path`: Path to the configuration file.
  - `vectordb_provider`: The vector store provider to use for the documents (Chroma or Qdrant).
  - `persist_directory`: The directory to store the vector store.
  - `splitting_method`: The method to split the documents into chunks (semantic or "recursive").
  - `path`: The path of the root folder of the documents you want to include in the index (will be explored recursively).
  - `process_log_file`: Log file to track processed documents.

- **Knowledge Graph Index Parameters**
  - `build_knowledge_graph`: Enable or disable building the knowledge graph when initializing the database.
  - `nb_nodes`: Number of nodes to retrieve from the knowledge graph when querying.
  - `allow_kg_retrieval`: Enable or disable retrieval of information from the knowledge graph during generation.

- **Chunking Parameters** (only for recursive splitting, which is not used anymore)
  - `chunk_size`: The chunk size for the documents in terms of the number of characters.
  - `chunk_overlap`: The overlap between chunks in terms of the number of characters.
  - `semantic_threshold`: The threshold for the semantic similarity between two chunks.
  - `chunking_embedding_model`: Embedding model to use for chunking.

- **Search Parameters**
  - `nb_chunks`: The number of chunks to retrieve from the database.
  - `search_type`: The type of search to use for the documents (similarity or exact).
  - `hybrid_search`: Enable or disable hybrid search (keyword + similarity).
  - `length_threshold`: The threshold for the length of the chunks (number of words).
  - `auto_hybrid_search`: Enable or disable the ability for the model to find the keyword itself.
  - `use_multi_query`: Enable or disable multi-query.
  - `advanced_hybrid_search`: Enable or disable advanced hybrid search (merge retrievers + routing if activated).
  - `alpha`: Alpha parameter for the Advanced Hybrid Search.
  - `enable_routing`: Enable or disable routing of queries to lexical/semantic search.
  - `data_sources`: Dictionary of data sources.

- **Search Filters**
  - `filter_on_length`: Filter the chunks based on length (number of words).
  - `enable_source_filter`: Enable or disable source filtering.
  - `word_filter`: Filter the results based on a keyword.
  - `source_filter`: Filter the results based on the source.
  - `source_filter_type`: Type of source filter (include or exclude).
  - `field_filter`: Field to filter on (source or length).
  - `field_filter_type`: Type of field filter (include or exclude).

- **Embedding Models Parameters**
  - `embedding_model`: Embedding model to use for vector store creation, retrieval, and semantic chunking.
  - `sparse_embedding_model`: Embedding model to use for sparse retrieval.
  - `sparse_embedding_size`: Size of the sparse embedding model.

- **Re-ranker Parameters**
  - `reranker_token_target`: Number of tokens to reach for the re-ranker.
  - `token_compression`: Enable or disable token compression.
  - `nb_rerank`: Number of documents to re-rank.
  - `use_reranker`: Enable or disable the re-ranker.
  - `reranker_model`: Re-ranker model to use.

- **Auto-Merging Parameters**
  - `auto_merging`: Enable or disable automatic merging of chunks.
  - `auto_merging_threshold`: Threshold for merging chunks (number of concomitant sources).

- **Generation Parameters**
  - `llm_provider`: Provider of the LLM model ('huggingface', 'ollama', or 'groq').
  - `model_name`: Model name.
  - `cot_enabled`: Enable or disable the Chain-of-Thought (COT) feature for the LLM answer.
  - `stream`: Enable or disable streaming of the generation.
  - `temperature`: Temperature for the generation.
  - `llm_token_target`: Number of tokens to reach for the LLM.
  - `save_answer`: Enable or disable saving the answer.
  - `fragmented_answer`: Enable or disable fragmented answers.
  - `prompt_language`: Language of the prompt.

- **Evaluation Parameters**
  - `top_k`: Evaluation is valid if the ground truth is in the top k chunks retrieved.
  - `evaluate_generation`: Enable or disable evaluation of the generation.
  - `answer_relevancy_llm`: LLM model to use for artificial queries.
  - `answer_relevancy_provider`: Provider of the LLM model for artificial queries.
  - `answer_relevancy_embedding`: Embedding model to use for artificial queries.
  - `metrics_list`: List of metrics to evaluate.

- **Entity/Relation Extraction Parameters**
  - `entity_model_name`: Model name for entity extraction.
  - `relation_model_name`: Model name for relation extraction.
  - `allowed_entities_path`: Path to allowed entities JSON file.
  - `allowed_relations_path`: Path to allowed relations JSON file.
  - `allowed_precise_relations_path`: Path to allowed detailed relations JSON file.
  - `entity_detection_threshold`: Threshold for entity detection.
  - `relation_extraction_threshold`: Threshold for relation extraction.
  - `disambiguate_threshold`: Threshold for disambiguation.

- **Community Summarization/Entity Description Parameters**
  - `description_model_name`: Model name for community summarization.
  - `description_llm_provider`: Provider of the LLM model for community summarization.

## 💻 Run the RAG with CLI Command

Before using a local LLM, you must run the following command to enable its use:

```bash
ollama serve
```

Then, pull the LLMs you want to use with the following command:

```bash
ollama pull llama3
```

**Note:** These steps are already performed during the Docker container building phase if you use the provided Dockerfile/Docker image, so you typically don't need to do this.

To ask a question and get an answer, run this script from the command line using the following command:

```bash
python scripts/launch_rag.py --question "Your question here" --config_file "Your configuration file here"
```

Example:

```bash
python scripts/launch_rag.py --question "What are the values of Euranova?" --config_file "config/config.yaml"
```

## 📊 Running the Evaluation

To run the evaluation, you need a configuration file in YAML format that specifies all the necessary parameters.

Once you have your configuration file ready, you can run the evaluation from the command line:

```bash
python evaluate_pipeline.py config.yaml
```

Or, more simply:

```bash
python evaluate_pipeline.py
```

**Note:** The evaluation dataset used is specific to the Chroma/Qdrant database and located in the persistent directory of the database (it's a JSON file).

The script will load the configuration from this file and then run the evaluation. The results of the evaluation will be printed to the console and also saved to a CSV file named `evaluation_results.csv` in the current directory.

## 🏃‍♂️ Running Parallel Evaluation

To run evaluations in parallel using multiple configuration files, you can use the `comparative_evaluation.py` script. This script will look for all YAML configuration files in a specified directory and run evaluations for each one.

You can run the parallel evaluation from the command line as follows:

```bash
python comparative_evaluation.py path_to_config_directory
```

If no directory is specified, the script will default to `config/comparative_evaluation`.

The results of each evaluation will be printed to the console and logged in a file named `evaluation.log` in the current directory.

Replace `path_to_config_directory` with the path to your directory containing multiple configuration files.

## 🗂️ Data Scraping

### Launch Main Scraping

The script `main_scrapper.py` allows you to run the different scraping scripts for GitLab, Mattermost, and Happeo sequentially. Here's how to use it:

1. **Set Environment Variables**

   Ensure you have the necessary environment variables defined in your `.env` file:

   - `GITLAB_URL`: The URL of your GitLab instance.
   - `PRIVATE_TOKEN`: Your GitLab private token.
   - `URL_MATTERMOST`: The URL of your Mattermost instance.
   - `MATTERMOST_TOKEN`: Your Mattermost access token.
   - `HAPPEO_URL`: The URL of your Happeo instance.
   - `HAPPEO_TOKEN`: Your Happeo access token.

2. **Run the Script**

   Execute the `main_scrapper.py` script using the following command:

   ```bash
   python scripts/main_scrapper.py [--gitlab_path <directory_path>] [--allow_parsing <true|false>] [--mattermost_path <directory_path>] [--happeo_path <directory_path>]
   ```

#### Optional Arguments

- `--gitlab_path`: The directory where GitLab README files will be saved (default: `data_test/gitlab`).
- `--allow_parsing`: A boolean flag to allow parsing of GitLab README files (default: `false`).
- `--mattermost_path`: The directory where Mattermost messages will be saved (default: `data_test/mattermost`).
- `--happeo_path`: The directory where Happeo pages will be saved (default: `data_test/happeo`).

#### Examples

- Run the script with default parameters:

  ```bash
  python scripts/main_scrapper.py
  ```

- Run the script with parsing of GitLab README files:

  ```bash
  python scripts/main_scrapper.py --allow_parsing true
  ```

- Run the script and save GitLab README files to a custom directory:

  ```bash
  python scripts/main_scrapper.py --gitlab_path custom_directory
  ```

- Run the script and save Mattermost messages to a custom directory:

  ```bash
  python scripts/main_scrapper.py --mattermost_path ./custom/path/
  ```

- Run the script and save Happeo pages to a custom directory:

  ```bash
  python scripts/main_scrapper.py --happeo_path ./custom/path/
  ```

This script performs the following:

- Launches the scraping of GitLab project README files.
- Launches the scraping of Mattermost channel messages.
- Launches the scraping of Happeo pages.
- Saves the scraped data to the specified directories.

⚠️ **It is recommended to run this script during the night or over the weekend because it can take a long time to scrape all the data.**

### Launch GitLab Scraping

To scrape README files from GitLab repositories:

1. **Set Environment Variables**

   Ensure you have the necessary environment variables set in your `.env` file:

   - `GITLAB_URL`: The URL of your GitLab instance.
   - `PRIVATE_TOKEN`: Your GitLab private token.

2. **Run the GitLab Scraping Script**

   ```bash
   python scripts/gitlab_scrapping.py [--allow_parsing <true|false>] [--path <directory_path>]
   ```

#### Optional Arguments

- `--allow_parsing`: A boolean flag to allow parsing of README files (default: `true`).
- `--path`: The directory where README files will be saved (default: `data/gitlab`).

#### Examples

- Run the script with default settings:

  ```bash
  python scripts/gitlab_scrapping.py
  ```

- Run the script without parsing README files:

  ```bash
  python scripts/gitlab_scrapping.py --allow_parsing false
  ```

- Run the script and save README files to a custom directory:

  ```bash
  python scripts/gitlab_scrapping.py --path custom_directory
  ```

### Launch Mattermost Scraping

⚠️⚠️ The Mattermost scraping script is currently not working because you need a cookie file in src/scrapping/cookies_happeo.pkl, you have to loggin once to obtain these cookies but the script to do that (obtain_cookies.py) doesn't work any more and i have no idea why⚠️⚠️


To scrape messages from Mattermost channels:

1. **Set Environment Variables**

   Ensure you have the necessary environment variables set in your `.env` file:

   - `URL_MATTERMOST`: The URL of your Mattermost instance.
   - `MATTERMOST_TOKEN`: Your Mattermost access token.

2. **Run the Mattermost Scraping Script**

   ```bash
   python scripts/mattermost_scrapping.py [--save_path <directory_path>]
   ```

#### Optional Arguments

- `--save_path`: The directory where the scraped data will be saved (default: `./data/mattermost_2.0/`).

#### Examples

- Run the scraping script with the default save path:

  ```bash
  python scripts/mattermost_scrapping.py
  ```

- Run the scraping script with a custom save path:

  ```bash
  python scripts/mattermost_scrapping.py --save_path ./custom/path/
  ```

### Launch Happeo Scraping

To scrape pages from Happeo:

1. **Set Environment Variables**

   Ensure you have the necessary environment variables set in your `.env` file:

   - `URL_HAPPEO`: The URL of your Happeo instance.
   - `HAPPEO_TOKEN`: Your Happeo access token.

2. **Run the Happeo Scraping Script**

   ```bash
   python scripts/happeo_scrapping.py [--directory <directory_path>]
   ```

#### Optional Arguments

- `--directory`: The directory where the scraped data will be saved (default: `./data/pages/`).

#### Examples

- Run the scraping script with the default save path:

  ```bash
  python scripts/happeo_scrapping.py
  ```

- Run the scraping script with a custom save path:

  ```bash
  python scripts/happeo_scrapping.py --directory ./custom/path/
  ```

## 📈 Evaluation (Without CLI Command)

The `evaluate_pipeline.py` script provides functions for evaluating the performance of the language model. The `init_evaluation_dataset` function generates an evaluation dataset by sampling random chunks from the Chroma database and generating artificial questions for each chunk. The `launch_eval` function loads the evaluation dataset and iterates over the examples, performing RAG with each query and counting the times when the answer is correct.

The `comparative_evaluation.py` script performs comparative evaluation of different methods based on a set of predefined config files. It iterates over each config file, loads the configuration, and launches the evaluation using the `launch_eval` function from `evaluate_pipeline.py`.

## 📄 Document Loading (Without CLI Command)

The `retrieval_utils.py` script provides functions for loading documents into the Chroma database. The `directory_to_vectorstore` function loads documents from a specified directory into the Chroma database. The `delete_chroma` function deletes the current Chroma/Qdrant database.