# 🚀 Project Description

Prepare to enter a new era of AI assistance! 🚀 This project isn't just a chatbot; it's a **cognitive revolution** 🧠, a **personal assistant supercharged with digital steroids** that far surpasses the capabilities of ChatGPT! 💪

Imagine a **Retrieval Augmented Generation (RAG)** 📚 system that draws its power from **all** your documents and the vast expanse of available information, all accessible in the blink of an eye from your computer. 💻

But wait, that's just the beginning! 🤩 Our chatbot is a true **Swiss Army knife of AI**, packed with functionalities that will leave you speechless:

*   **Lightning-Fast Audio Transcription** ⚡: Endless meetings? 🥱 Lengthy YouTube videos? 😴 Our AI transcribes them at the speed of light, faster than you can say "artificial intelligence"! ✨
*   **Limitless Virtual Brain** 🧠: Graphs, images, captivating LinkedIn posts, audio transcriptions, PDFs... 📊🖼️📝 Absolutely **every imaginable format** is analyzed and stored in a virtual brain that expands infinitely, ready to answer your most complex queries.
*   **Master of Your Emails** 📧: Say goodbye to email overload! 📥 Our chatbot dives into your inbox, answers your questions, and frees you from the burden of email management.
*   **Notion Page Creator** 📑: Need to organize your ideas? Our AI generates structured and elegant Notion pages in an instant, allowing you to focus on what matters most.
*   **Ultimate LLM Flexibility** 😎: Llama 3, Gemini, GPT-4o... 🦙♊🤖 No matter the language model, our system **supports them all**! You have the power to choose the best tool for each task, offering unprecedented flexibility in the world of AI.
*   **Breathtaking Speed and Performance** 🏎️: Forget endless waiting times. Our chatbot is optimized for **ultra-fast** processing and response speed, delivering **state-of-the-art** performance that redefines efficiency.

This project is much more than just a tool; it's your **intelligent companion**, your **ultimate assistant**, ready to propel you to new heights of productivity and creativity. 🚀

**Join the revolution and unleash the unlimited potential of AI!** 🔥🔥🔥


<img src="assets/logo_v4.png" alt="Logo" width="300" height="300">

## 🔑 Key Features

-**Support All Free LLMs providers !**: Our system supports all free tiers LLM providers: Forget the expensive OpenAI API, we support lightning fast Llama3.3 calls with Groq, Gemini 2.0 access through google ai studio, GPT4o through google marketplace, everything is there for free !
- **Simple Question Answering**: Uses the HuggingFace language model to answer questions based on the content of documents loaded into the Chroma database.
- **Advanced Question Answering**: Can answer complex questions by decomposing them into sub-questions and aggregating answers from multiple sources.
- **Evaluation**: Provides scripts for evaluating the performance of the language model.
- **Document Loading**: Functions for loading documents into the Chroma database.
- **Streamlit Interface**: Offers a Streamlit interface for user interaction with the application.
- **Audio Transcription**: Transcribes audio files to text in a blink of an eye using an optimized implementation of the WhisperV3 model through Groq API and PyAnnote speaker diarization model.
- **Optical Character Recognition (OCR)**: Extracts text from images using a state-of-the-art OCR model or Vision LLMs, extracts information from charts and tables, and generates structured equivalents in json or markdown format.
- **Mail Assistant**: Reads and answers emails, extracts information, and performs actions based on the content of the emails.


## 🛠️ Installation: Choose Your Own Adventure! 🚀

Ready to unleash the power of our revolutionary AI assistant? 🤩 You have two epic paths to choose from:

### **Option 1: Docker - The Containerized Kingdom** 🐳

For those who love the streamlined elegance of Docker, this is your path to AI glory! 🛡️

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

    *Optional: Build from scratch (not usually needed) - for the hardcore builders!* 💪

    ```bash
    docker build --no-cache -t llms-rag/demo:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
    ```

4. **Run the Docker Container**

    *Specify the device you want to use with the `--gpus` flag: like a true AI warrior!* ⚔️

    ```bash
    docker run -d --rm --gpus '"device=0"' -v $(pwd):/home/user/llms --name=llms llms-rag/demo:latest
    ```

    *For a permanent run: because this AI deserves to be eternal!* ✨

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

    *If Streamlit asks you to enter your email, just press Enter to bypass this step like a ninja!* 🥷

    *From this container, follow the instructions below to run the project.*

### **Option 2: Local Installation - Unleash the Power of `uv`!** ⚡

Prefer a more hands-on approach? Want to harness the blazing-fast speed of `uv`, the revolutionary Python package manager and environment gestion tool? Then this is your path! 🔥

1. **Clone the Repository**

    ```bash
    git clone [repository_url]
    ```

2. **Navigate to the Project Root Directory**

    ```bash
    cd llms-rag
    ```

3. **Install Dependencies with `uv`** - Experience the speed! 🤯 `uv` will automatically detect and install dependencies from your `pyproject.toml` file.
     If you don't already have `uv` installed, install it first:
    ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    Then install the dependencies:
    ```bash
    uv pip install -e .
    ```

4. **Run the Streamlit Application**

    ```bash
    streamlit run streamlit_app.py
    ```

    *If Streamlit asks you to enter your email, just press Enter to skip this step.* 😎

**No matter which path you choose, prepare for an AI experience like no other!** 🎉


## 📦 Setup

First, you need to fill the `.env` file with the following variables (not all of them are mandatory for running the project):

```bash
GROQ_API_KEY="..."
LANGCHAIN_API_KEY="..."
LANGCHAIN_END_POINT="https://api.smith.langchain.com"
LANGCHAIN_TRACING_v2="true" #or false
HUGGINGFACEHUB_API_TOKEN="..." # Deprecated, use HUGGINGFACE_TOKEN instead
HUGGINGFACE_TOKEN="..."
GITLAB_URL='https://code....'
GITLAB_TOKEN="..."
STREAMLIT_URL='http://10.0.13.231:8501'
ALLOWED_FORMATS=["pdf", "docx", "xlsx", "html", "pptx", "txt", "md"]
URL_MATTERMOST="..." # Deprecated
MATTERMOST_TOKEN="..." # Deprecated
EMAIL="..."
PASSWORD="..."
SAMBANOVA_API_KEY="..."
GITHUB_TOKEN="..."
IMAP_SERVER="..."
EMAIL_ADDRESS="..."
EMAIL_PASSWORD="..."
SUNO_COOKIE="..."
PYTHONPATH="..."
LINKEDIN_USERNAME="..."
LINKEDIN_PASSWORD="..."
NOTION_API_KEY="..."
GOOGLE_API_KEY="..."
GOOGLE_SERVICE_MAIL="..."
VERTEX_API_KEY="..."
```

- **GROQ_API_KEY**: Obtain from the [Groq Cloud website](https://console.groq.com/keys).
- **LANGCHAIN_API_KEY**: Obtain from the [LangChain website](https://smith.langchain.com/settings).
- **LANGCHAIN_END_POINT**: Endpoint for the LangChain API (default: `https://api.smith.langchain.com`).
- **LANGCHAIN_TRACING_v2**: Enable or disable LangChain tracing (set to `true` or `false`).
- **HUGGINGFACEHUB_API_TOKEN**: **Deprecated**. Use `HUGGINGFACE_TOKEN` instead.
- **HUGGINGFACE_TOKEN**: Obtain from the [HuggingFace website](https://huggingface.co/settings/tokens) (create a new token with "write" permissions).
- **GITLAB_URL**: URL of your GitLab instance (e.g., `https://code.euranova.eu/`).
- **PRIVATE_TOKEN**: **Deprecated**. Use `GITLAB_TOKEN` instead.
- **GITLAB_TOKEN**: Personal Access Token for GitLab. Go to your GitLab settings -> Access Tokens, and create a new token with the `api` scope.
- **STREAMLIT_URL**: URL of the Streamlit application.
- **ALLOWED_FORMATS**: Specifies the allowed file formats for documents (no other formats are supported).
- **URL_MATTERMOST**: **Deprecated**.
- **MATTERMOST_TOKEN**: **Deprecated**.
- **EMAIL**: Your email address (used for general purposes within the application).
- **PASSWORD**: Your email password (used for general purposes within the application).
- **SAMBANOVA_API_KEY**: Obtain from the SambaNova website.
- **GITHUB_TOKEN**: Personal Access Token for GitHub. Go to your GitHub settings -> Developer settings -> Personal access tokens, and create a new token with the necessary permissions (e.g., `repo`, `read:org`).
- **IMAP_SERVER**: The address of the IMAP server of your email adress (e.g., `imap.centrale-med.fr`).
- **EMAIL_ADDRESS**: Your email address (used for email functionalities).
- **EMAIL_PASSWORD**: Your email password (used for email functionalities).
- **SUNO_COOKIE**: Your Suno AI cookie. Obtain it by inspecting your browser's network requests while using Suno AI.
- **PYTHONPATH**:  Set this to the root directory of the project if you encounter import issues.
- **LINKEDIN_USERNAME**: Your LinkedIn username.
- **LINKEDIN_PASSWORD**: Your LinkedIn password.
- **NOTION_API_KEY**: Your Notion API key. Obtain it from your Notion workspace settings.
- **GOOGLE_API_KEY**: Your Google Cloud API key. Obtain it from the Google Cloud Console.
- **GOOGLE_SERVICE_MAIL**: Your Google service account email.
- **VERTEX_API_KEY**: Your Vertex AI API key. Obtain it from the Google Cloud Console.

Once you have filled the `env` file with the correct values, rename it to `.env`.

## 🚀 Quickstart

Here's a summary of the most important steps to start using the project:

1. **Ensure the Docker Container is Running / venv is setup**

   Make sure the Docker container is up and running or that the venv is correctly setup.

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
   streamlit run scripts/streamlit_app.py
   ```

2. **Use the Interface**

   - Use the parameters in the sidebar to configure the assistant as you wish.
   - Type your question in the user input field.
   - Click the `Ask` button to get an answer.

## 📁 Directory Structure

The project is organized as follows:

- **Root Directory**

  - `README.md`: The main documentation file for the project.
  - `data/vector_stores/`: Directory containing the different vector stores for the documents, their log files, and respective evaluation datasets. The name of the vector store includes the vector store provider, type of chunking, and the embedding model used.
  - `config/config.yaml`: Configuration file for the project. All parameters for evaluation and retrieval are stored here.
  - `.env`: File containing the environment variables for the project (API keys, etc.).

- **`src` Directory**

  - `retrieval_utils.py`: Functions for loading documents into the vector store.
  - `data/`: Directory containing all the data of the user's documents.
  - `evaluate_pipeline.py`: Functions for evaluating the performance of the language model.
  - `generation_utils.py`: Implementation of the question-answering model/RAG pipeline and associated functions.
  - `utils.py`: Various utility functions used throughout the project.
  - `knowledge_graph.py`: Functions to create a knowledge graph from the documents.
  - `LLM.py`: Implementation of all language models available for free on the market, at the base of the project and the way to access the different LLMs implemented (through Ollama or LangChain/Groq API, ect..).


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

The `fill_database` command is used to initialize the Qdrant vector store. It takes as an argument the name of the YAML configuration file that contains the necessary parameters for creating the vector store if it does not already exist and fills it with the documents from the specified directory. Documents already in the database will not be added again.

Here's how you can use this command:

```bash
python src/vectorstore_utils_v4.py config.yaml
```


**Note:** Replace `config/config.yaml` with the path to your configuration file.

### Command Details

The `fill_database` command performs the following operations:

1. Reads the YAML configuration file specified as an argument.
2. Checks if the configuration file exists; if not, it displays an error message and terminates.
3. Loads the configuration parameters from the YAML file.
4. Calls the `directory_to_vectorstore` function with the loaded configuration parameters to create the Chroma/Qdrant database using the specified parameters.

Okay, here's the revised configuration parameters section, reflecting the changes in your updated config file, removing deprecated parameters, adding new ones, and formulated in a more understandable and engaging way:

### ⚙️ Configuration: Fine-Tune Your AI Brain! 🧠

This section details the powerful configuration options, allowing you to customize the behavior of our AI assistant like a true maestro! 🎻 These parameters, defined in your `config/config.yaml` file, are the key to unlocking its full potential.

**Here’s a breakdown of the different settings:**

- **🗄️ Database & Indexing Setup:**
    -   `config_path`: The path to this very configuration file. Think of it as the AI's control panel address.
    -   `vectordb_provider`: Choose your vector database warrior: `Qdrant` (fast and furious) or `Faiss` (classic and reliable).
    -   `persist_directory`: Where the AI stores its memories (the vector database). It's like the library where your documents are indexed and kept safe.
    -   `collection_name`: The name of the vector collection in the database.
    -   `path`: The path to the root folder containing all the documents the AI will learn from. This is where the magic begins! ✨
    -   `process_log_file`: A log file to keep track of which documents have already been processed, preventing any double processing.
    -   `splitting_method`: How the AI should break up documents into manageable chunks: `"constant"` (keeps the chunks of similar sizes), `"semantic"` (chunks based on the content semantic).
    
- **🧠 Knowledge Graph Enhancement:**
    -   `build_knowledge_graph`: Enable this to build a knowledge graph (or disable it), a structured representation of the relationships between concepts.
    -   `nb_nodes`: When querying, how many nodes should be retrieved from the knowledge graph (similar to the number of chunks).
    -   `allow_kg_retrieval`: Enable information retrieval from the knowledge graph during answer generation.
    -   `kg_target_context`: Number of tokens to target for the knowledge graph retrieval.
    -   `min_relations_pct`: Minimum number of relations required for the entity to be a node in the community description.
    -   `min_descriptions_pct`: Minimum number of descriptions required for the entity to be a node in the community description.
    -   `min_communities_pct`: Minimum number of communities required for the entity to be a node in the community description.

-   **✂️ Chunking Strategy (for constant splitting method):**
    -   `chunk_size`: The size of each document chunk in number of characters.
    -   `chunk_overlap`: The overlap size between chunks in number of characters.
    -   `semantic_threshold`: The similarity threshold between two chunks when splitting using semantic splitting.
    -    `chunking_embedding_model`: The embedding model used for chunking if using semantic splitting.
   
- **🗂️ Database Cloning (Advanced Users):**
    - `clone_database`: If you want to clone an existing database to initialize your new one, set this to true, if not, disable this parameter.
    - `clone_persist`: Path to the directory where the existing index is stored.
    - `clone_embedding_model`: Embedding model of the index to be cloned.

-   **🔍 Search & Retrieval Customization:**
    -   `nb_chunks`: How many document chunks should the AI retrieve when searching.
    -   `search_type`:  How should the AI perform the search? `"similarity"` (semantic search) or `"exact"` (keyword search).
    -   `hybrid_search`: Combine the power of keyword search with semantic search for better results.
    -   `length_threshold`: Filter chunks that are too short.
    -  `use_multi_query`: Enable multiple queries from the user input.
    -  `advanced_hybrid_search`: Enable or disable advanced hybrid search (merging the results of different retrievers and routing).
    -  `deep_search`: Enable or disable advanced RAG answer (Multi step answer generation).
    -  `alpha`: A parameter for the Advanced Hybrid Search, defining the weight given to the semantic retriever and lexical retriever.
    -   `enable_routing`: Route user queries to either lexical or semantic search based on the query type.
    - `suggestions_enabled`: Enable or disable the suggestions of the query.
    - `use_history`: Enable the use of the chat history to answer the user queries.
    - `chat_history`: Chat history to use for the queries.
    -   `data_sources`: A dictionary of different data sources with their description, helping the model to pick the appropriate sources for the context.

-  **🔎 Search Filters**
    -  `use_autocut`: Automatically cut the chunks depending on the length threshold (if there is a big downward tendency, cut).
    -  `autocut_beta`: Beta parameter for the autocut.
    -  `filter_on_length`: Filter chunks based on their length.
    -  `enable_source_filter`: Filter the chunks depending on their source.
    -  `word_filter`: Filter the chunks containing a particular word.
    -   `source_filter`: List of specific sources to be included or excluded.
    -   `source_filter_type`: Should the AI include or exclude the sources specified in `source_filter`? ($eq for include and $neq for exclude)
    -   `field_filter`:  Filter based on specific fields (source or length).
    -   `field_filter_type`: How should the filtering occur (include or exclude). ($eq for include and $neq for exclude)

-   **✨ Embedding Power:**
    -   `embedding_model`: The embedding model for creating vector representations of your text data.
    -  `dense_embedding_size`: Size of the dense embedding model.
    -   `sparse_embedding_model`: The embedding model used for sparse retrieval.
    -  `sparse_embedding_size`: Size of the sparse embedding model.

-   **🥇 Reranking Refinement:**
    -  `reranker_token_target`: Number of tokens to target for the re-ranker.
    -  `token_compression`: Compress the number of tokens of the chunks in order to respect `reranker_token_target`.
    -   `nb_rerank`: Number of documents to keep from the reranking process.
    -   `use_reranker`: Enable the reranker, a secondary model that re-orders the documents based on their relevance.
    -   `reranker_model`: The re-ranking model that will be used.

-  **🔄 Auto-Merging Magic**
    -  `auto_merging`: Enables the automatic merging of chunks (if multiple chunks coming from the same source).
    -  `auto_merging_threshold`: The threshold for the auto merging.

-   **🤖 Generation & Response Settings:**
    -   `llm_provider`:  Choose your LLM provider (like `groq`, `github`, or `sambanova`).
    -    `model_name`: The specific language model to use (make sure it matches the selected provider).
    -    `models_dict`: Dictionnary of models linked with their provider.
    -    `fr_rag_prompt_path`: Path to the file containing the RAG prompt in French.
    -    `en_rag_prompt_path`: Path to the file containing the RAG prompt in English.
    -    `vllm_model_name`: LLM model to use with the VLLM interface.
    -    `vllm_provider`: Provider for the VLLM interface.
    -   `cot_enabled`: Unleash the power of Chain-of-Thought reasoning for more sophisticated answers (or disable if you don't need it).
    -   `stream`: Whether the answers should be printed token by token or all at once.
    -   `temperature`: Adjusts the creativity of the LLM responses (lower values = more deterministic).
    -   `llm_token_target`: The number of tokens that we give to the llm. If set to 0, this feature is disabled. If autocut is enabled, this feature is not used.
    -   `save_answer`: Save or not the answer.
    -   `prompt_language`: Language of all the prompts used by the model.

- **📊 Evaluation Metrics:**
    -   `top_k`: The retrieval is considered successful if the correct document is in the top k results.
    -   `evaluate_generation`: Enable evaluation of the whole pipeline (or just the retrieval if disabled).
    -   `answer_relevancy_llm`:  The LLM model for generating artificial queries for retrieval evaluation.
    -   `answer_relevancy_provider`: The provider of the LLM model for artificial queries.
    -   `answer_relevancy_embedding`: Embedding model to use for artificial queries.
    -   `metrics_list`: The list of metrics to compute.

- **🎭 Entity/Relation Extraction:**
    -   `entity_model_name`: The model to extract named entities from the text.
    -   `relation_model_name`: The model to identify relationships between entities.
    -   `allowed_entities_path`: Path to the file containing the allowed entities types.
    -   `allowed_relations_path`: Path to the file containing the allowed relations types.
    -   `allowed_precise_relations_path`: Path to the file containing the allowed detailed relations types.
    -   `entity_detection_threshold`: Minimum confidence score for an entity to be detected.
    -   `relation_extraction_threshold`: Minimum confidence score for a relation to be detected.
    -   `disambiguate_threshold`: Minimum similarity score for disambiguation.

-  **🗣️ Community & Entity Descriptions:**
    -   `description_model_name`: The LLM model for creating a description for communities and entities.
    -  `description_llm_provider`: Provider of the LLM model for the entity descriptions.

- **🎯 Actions Classifier**
    - `actions_dict`: Dictionary of the actions that the user can ask, with their description.
    
- **🤖 Agentic RAG parameters**
    - `query_breaker_model`: The LLM model used for decomposing the queries.
    - `query_breaker_provider`: The provider of the LLM model used for decomposing the queries.
    
- **🧠 Intent Classifier Parameters:**
     - `query_classification_model`: LLM model to classify the intent of the user query.
     - `query_classification_provider`: Provider of the LLM model for the intent classifier.

With these settings, you can mold the AI to fit your needs perfectly! 🎉 Don't hesitate to experiment and see what magic you can create! 🪄



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
python scripts/launch_rag.py --question "What says my last medical record ?" --config_file "config/config.yaml"
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