[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yanis-labeyrie-67b11b225/)
[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=flat&logo=instagram&logoColor=white)](https://www.instagram.com/aiandcivilization/)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=youtube&logoColor=white)](https://www.youtube.com/@aiandcivilization)

# 🚀 The Ultimate AI assistant ! 🚀

Get ready to experience the future of AI assistance! This isn't just another chatbot; it's a **cognitive powerhouse** 🧠, a **digital assistant on hyperdrive** that will leave ChatGPT in the dust! 💪

Imagine a **Retrieval Augmented Generation (RAG)** 📚 system that taps into **all** of your documents and the limitless depths of the internet, all accessible from your computer in the blink of an eye. 💻

But that's just the opening act! 🤩 Our chatbot is a true **AI Swiss Army knife**, brimming with incredible features that will absolutely blow your mind:

*   **Blazing-Fast Audio Transcription** ⚡: Drowning in meetings? 🥱 Endless YouTube videos? 😴 Our AI transcribes them at warp speed, faster than you can say "artificial intelligence"! ✨
*   **Limitless Virtual Brain** 🧠: Graphs, images, captivating LinkedIn posts, audio transcriptions, PDFs... 📊🖼️📝 Absolutely **every imaginable format** is analyzed and stored in a virtual brain that expands infinitely, always ready with the answer to your most complex questions.
*   **Master of Your Emails** 📧: Say goodbye to email chaos! 📥 Our chatbot dives into your inbox, answers your queries, and frees you from the shackles of email management.
*   **Notion Page Alchemist** 📑: Need to organize your brilliant ideas? Our AI crafts structured and elegant Notion pages in an instant, letting you focus on what truly matters.
*   **Ultimate LLM Flexibility** 😎: Llama 3, Gemini, GPT-4o... 🦙♊🤖 No matter the language model, our system **supports them all**! You have the power to choose the perfect tool for every task, giving you unprecedented freedom in the world of AI.
*   **Mind-Blowing Speed and Performance** 🏎️: Forget endless loading times. Our chatbot is optimized for **lightning-fast** processing and response, delivering **state-of-the-art** performance that redefines efficiency.

This project isn't just a tool; it's your **intelligent partner**, your **ultimate assistant**, ready to catapult you to new heights of productivity and creativity. 🚀

**Join the revolution and unleash the limitless potential of AI!** 🔥🔥🔥

<img src="assets/logo_v5.png" alt="Logo" width="300" height="300">

## 🔑 Key Features

- **Supports All Free LLM Providers!**: Our system seamlessly supports all free-tier LLM providers. Forget expensive OpenAI APIs; enjoy lightning-fast Llama3 calls with Groq, Gemini 2.0 access through Google AI Studio, and GPT4o via Google Marketplace—all completely free!
- **Effortless Question Answering**: Employs the HuggingFace language model to answer questions based on the wealth of documents loaded into the Chroma database.
- **Advanced Question Answering**: Tackles even the most complex questions by breaking them down into sub-questions and combining answers from multiple sources for a comprehensive response.
- **Robust Evaluation Framework**: Provides scripts for thoroughly evaluating the performance of the language model.
- **Streamlined Document Loading**: Offers functions to seamlessly load documents into the Chroma database.
- **Intuitive Streamlit Interface**: Delivers a user-friendly Streamlit interface for effortless interaction with the application.
- **Instant Audio Transcription**: Transcribes audio files to text in the blink of an eye using an optimized implementation of the WhisperV3 model through the Groq API and the PyAnnote speaker diarization model.
- **Advanced OCR Capabilities**: Extracts text from images using a state-of-the-art OCR model or Vision LLMs. Extracts information from charts and tables and generates structured equivalents in JSON or Markdown format.
- **Intelligent Mail Assistant**: Reads and responds to emails, extracts key information, and performs actions based on the content of the emails.

## 🛠️ Installation: Choose Your Path to AI Mastery! 🚀

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

### **Option 2: Local Installation - Harness the Power of `uv`!** ⚡

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

First, populate the `.env` file with the following variables (not all are mandatory):

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
- **IMAP_SERVER**: The address of the IMAP server of your email address (e.g., `imap.centrale-med.fr`).
- **EMAIL_ADDRESS**: Your email address (used for email functionalities).
- **EMAIL_PASSWORD**: Your email password (used for email functionalities).
- **SUNO_COOKIE**: Your Suno AI cookie. Obtain it by inspecting your browser's network requests while using Suno AI.
- **PYTHONPATH**: Set this to the root directory of the project if you encounter import issues.
- **LINKEDIN_USERNAME**: Your LinkedIn username.
- **LINKEDIN_PASSWORD**: Your LinkedIn password.
- **NOTION_API_KEY**: Your Notion API key. Obtain it from your Notion workspace settings.
- **GOOGLE_API_KEY**: Your Google Cloud API key. Obtain it from the Google Cloud Console.
- **GOOGLE_SERVICE_MAIL**: Your Google service account email.
- **VERTEX_API_KEY**: Your Vertex AI API key. Obtain it from the Google Cloud Console.

Once you have filled the `env` file with the correct values, rename it to `.env`.

## 🚀 Quickstart

Here's a concise guide to get you up and running:

1.  **Ensure the Docker Container is Running / venv is setup**

    Confirm the Docker container is running or that the venv is correctly setup.

2.  **Verify the Project Root**

    Ensure you are at the root of the project.

3.  **Prepare the Vector Store**

    *   Verify that a vector store containing your data exists in the `data/vector_stores/` directory and that its name is specified in the configuration file under the `persist_directory` parameter.
    *   If it does not exist, you will need to create it (requires GPU!). To create it, ensure you have a `data` directory containing all the data you want to include in the index at the root of the project. Ensure the path to this directory is specified in the configuration file under the `path` parameter.
    *   Then run the following command to fill the database:

        ```bash
        python scripts/fill_database.py
        ```

4.  **Run the Streamlit Application**

    After completing the previous steps, run the Streamlit application:

    ```bash
    streamlit run scripts/streamlit_app.py
    ```

5.  **Interact with the RAG Chatbot**

    Open the Streamlit application using the local link and engage with the RAG chatbot through the intuitive interface.

**Note:** Initial loading of models may cause delays on first run.

## 🖱️ How to Use the Streamlit Interface

1.  **Launch the Streamlit Application**

    Run the application using:

    ```bash
    streamlit run scripts/streamlit_app.py
    ```

2.  **Use the Interface**

    *   Use the sidebar parameters to tailor the assistant to your needs.
    *   Enter your question in the user input field.
    *   Click the `Ask` button to receive your answer.

## 📁 Directory Structure

The project is structured for easy navigation:

-   **Root Directory**

    *   `README.md`: The project's main documentation file.
    *   `data/vector_stores/`: Stores the different vector databases for your documents, their log files, and respective evaluation datasets. The vector store's name includes the provider, chunking type, and embedding model used.
    *   `config/config.yaml`: The central configuration file for the project. All parameters for evaluation and retrieval are stored here.
    *   `.env`: Contains the environment variables for the project (API keys, etc.).

-   **`src` Directory**

    *   `retrieval_utils.py`: Functions for loading documents into the vector database.
    *   `data/`: Holds all user document data.
    *   `evaluate_pipeline.py`: Contains functions for evaluating the language model's performance.
    *   `generation_utils.py`: The core implementation of the question-answering model/RAG pipeline and associated functions.
    *   `utils.py`: Assorted utility functions used throughout the project.
    *   `knowledge_graph.py`: Functions for creating a knowledge graph from the documents.
    *   `LLM.py`: Implementation of all freely available language models and the methods for accessing them. (through Ollama, LangChain, Groq API, etc).

-   **`scripts` Directory**

    *   `launch_rag.py`: CLI command for launching the RAG model.
    *   `fill_database.py`: CLI command for populating/initializing the Chroma database from a directory.
    *   `streamlit_app.py`: Houses the Streamlit interface for the RAG chatbot and launches it.
    *   `evaluation_results.csv`: CSV file storing the evaluation results with different settings/hyperparameters.
    *  `happeo_scrapping.py`: Implementation of functions to scrape Happeo pages.
    *   `mattermost_scrapping.py`: Implementation of functions to scrape Mattermost channels.
    *   `gitlab_scrapping.py`: Implementation of functions to scrape GitLab repositories' READMEs.
    *   `comparative_evaluation.py`: Script for performing comparative evaluations based on predefined config files.
    *   `evaluate_pipeline.py`: Functions for evaluating the language model's performance and the retrieval pipeline.
    *   `clustering_by_topic.py`: Script to cluster document chunks by topic using BERTopic and Qdrant.
    *   `eval_dataset_curation_streamlit.py`: Streamlit interface for generating evaluation datasets by sampling random chunks from the Chroma or Qdrant database and generating artificial questions for each chunk.

## 🔄 Fill the Qdrant Database

The `fill_database` command initializes the Qdrant vector store. It takes the path to the YAML configuration file and uses it to create the database if it doesn't exist and fills it with documents from the specified directory, skipping already existing documents in the database.

Here's how to use it:

```bash
python src/vectorstore_utils_v4.py config.yaml
```

**Note:** Replace `config/config.yaml` with the path to your configuration file.

### Command Details

The `fill_database` command performs the following actions:

1.  Reads the YAML configuration file specified as an argument.
2.  Verifies the configuration file exists; if not, it terminates with an error message.
3.  Loads the configuration parameters from the YAML file.
4.  Calls the `directory_to_vectorstore` function with the loaded parameters to create the Chroma/Qdrant database.

## ⚙️ Configuration: Fine-Tune Your AI Brain! 🧠

This section outlines the powerful configuration options, enabling you to customize your AI assistant. These parameters in your `config/config.yaml` file are key to unlocking its full potential.

**Here's a breakdown:**

-   **🗄️ Database & Indexing Setup:**
    *   `config_path`: Path to this configuration file.
    *   `vectordb_provider`: Select your vector database: `Qdrant` or `Faiss`.
    *   `persist_directory`: Storage for the vector database.
    *   `collection_name`: Name of the vector collection.
    *   `path`: Root folder containing the documents.
    *   `process_log_file`: Log file for tracking processed documents.
     * `splitting_method`: Choose between "constant" and "semantic" chunking.

- **🧠 Knowledge Graph Enhancement:**
    -   `build_knowledge_graph`: Enable this to build a knowledge graph.
    -   `nb_nodes`: Number of nodes to retrieve from the knowledge graph.
    -   `allow_kg_retrieval`: Enable information retrieval from the knowledge graph during answer generation.
    -   `kg_target_context`: Number of tokens to target for the knowledge graph retrieval.
    -   `min_relations_pct`: Minimum number of relations for an entity to be a node in the community description.
    -   `min_descriptions_pct`: Minimum number of descriptions for an entity to be a node in the community description.
    -   `min_communities_pct`: Minimum number of communities for an entity to be a node in the community description.

-   **✂️ Chunking Strategy (for constant splitting method):**
    *   `chunk_size`: Size of each document chunk.
    *   `chunk_overlap`: Overlap between chunks.
    *   `semantic_threshold`: The similarity threshold between two chunks when splitting using semantic splitting.
     *  `chunking_embedding_model`: The embedding model used for chunking if using semantic splitting.
-   **🗂️ Database Cloning (Advanced Users):**
    *   `clone_database`: Set to `true` to clone an existing database.
    *   `clone_persist`: Path to the directory of the existing index.
    *   `clone_embedding_model`: Embedding model of the index to be cloned.

-   **🔍 Search & Retrieval Customization:**
    *   `nb_chunks`: Number of document chunks to retrieve.
    *   `search_type`:  Search type: `"similarity"` or `"exact"`.
    *   `hybrid_search`: Combine keyword and semantic search.
     * `length_threshold`: Filter chunks that are too short.
    *   `use_multi_query`: Enable multiple queries from the user input.
    *  `advanced_hybrid_search`: Enable or disable advanced hybrid search.
     * `deep_search`: Enable or disable advanced RAG answer (Multi step answer generation).
    *  `alpha`: Parameter for Advanced Hybrid Search.
    *  `enable_routing`: Route queries to lexical or semantic search based on type.
    *  `suggestions_enabled`: Enable or disable query suggestions.
     *   `use_history`: Enable use of chat history.
     *  `chat_history`: Chat history to use for queries.
    *   `data_sources`: Dictionary of data sources and their descriptions.

-  **🔎 Search Filters**
    -  `use_autocut`: Automatically cut the chunks depending on the length threshold.
    -  `autocut_beta`: Beta parameter for the autocut.
    -  `filter_on_length`: Filter chunks based on their length.
     -  `enable_source_filter`: Filter the chunks depending on their source.
    -  `word_filter`: Filter the chunks containing a particular word.
    *   `source_filter`: List of sources to include or exclude.
    *   `source_filter_type`:  Include or exclude specified sources.
    *   `field_filter`: Filter based on specific fields (source or length).
    *   `field_filter_type`: How the filtering should occur (include or exclude).

-   **✨ Embedding Power:**
    *   `embedding_model`: Embedding model for text data.
      *  `dense_embedding_size`: Size of the dense embedding model.
    *   `sparse_embedding_model`: Embedding model for sparse retrieval.
    *  `sparse_embedding_size`: Size of the sparse embedding model.

-   **🥇 Reranking Refinement:**
     *  `reranker_token_target`: Number of tokens to target for the re-ranker.
    *  `token_compression`: Compress the number of tokens of the chunks in order to respect `reranker_token_target`.
     *   `nb_rerank`: Number of documents to keep after reranking.
    *   `use_reranker`: Enable the reranker.
    *   `reranker_model`: Reranking model to use.

-  **🔄 Auto-Merging Magic**
    -  `auto_merging`: Enables the automatic merging of chunks.
    -  `auto_merging_threshold`: The threshold for the auto merging.

-   **🤖 Generation & Response Settings:**
    *   `llm_provider`: LLM provider (like `groq`, `github`, or `sambanova`).
    *   `model_name`: Specific language model to use.
    *   `models_dict`: Dictionnary of models linked with their provider.
    *    `fr_rag_prompt_path`: Path to the file containing the RAG prompt in French.
    *    `en_rag_prompt_path`: Path to the file containing the RAG prompt in English.
     *   `vllm_model_name`: LLM model to use with the VLLM interface.
     *    `vllm_provider`: Provider for the VLLM interface.
    *   `cot_enabled`: Enable Chain-of-Thought reasoning.
    *   `stream`: Stream the answers token by token.
    *   `temperature`: Adjust the creativity of the LLM responses.
    *  `llm_token_target`: The number of tokens that we give to the llm. If set to 0, this feature is disabled. If autocut is enabled, this feature is not used.
    *  `save_answer`: Save or not the answer.
    *   `prompt_language`: Language of the prompts.

- **📊 Evaluation Metrics:**
    *   `top_k`: Retrieval is successful if the correct document is in the top k results.
     *   `evaluate_generation`: Enable evaluation of the whole pipeline.
    *   `answer_relevancy_llm`:  LLM model for generating artificial queries for retrieval evaluation.
    *   `answer_relevancy_provider`: Provider of the LLM model for artificial queries.
     *   `answer_relevancy_embedding`: Embedding model to use for artificial queries.
    *   `metrics_list`: List of metrics to compute.

- **🎭 Entity/Relation Extraction:**
    *   `entity_model_name`: Model for extracting entities.
    *   `relation_model_name`: Model for identifying relationships between entities.
    *   `allowed_entities_path`: Path to allowed entities types.
    *   `allowed_relations_path`: Path to allowed relations types.
    *    `allowed_precise_relations_path`: Path to the file containing the allowed detailed relations types.
    *   `entity_detection_threshold`: Minimum confidence for entity detection.
    *   `relation_extraction_threshold`: Minimum confidence for relation detection.
    *    `disambiguate_threshold`: Minimum similarity score for disambiguation.

-  **🗣️ Community & Entity Descriptions:**
     *  `description_model_name`: LLM model for creating a description for communities and entities.
     *  `description_llm_provider`: Provider of the LLM model for the entity descriptions.

-  **🎯 Actions Classifier**
    - `actions_dict`: Dictionary of the actions that the user can ask, with their description.
    
- **🤖 Agentic RAG parameters**
    - `query_breaker_model`: The LLM model used for decomposing the queries.
    - `query_breaker_provider`: The provider of the LLM model used for decomposing the queries.
    
- **🧠 Intent Classifier Parameters:**
     - `query_classification_model`: LLM model to classify the intent of the user query.
     - `query_classification_provider`: Provider of the LLM model for the intent classifier.

These settings empower you to tailor the AI to your exact requirements! 🎉 Don't hesitate to experiment and witness the magic! 🪄

## 💻 Run the RAG with CLI Command

Before using a local LLM, execute the following command:

```bash
ollama serve
```

Then, pull the desired LLMs with:

```bash
ollama pull llama3
```

**Note:** These steps are already handled during the Docker container building if you are using it.

To ask a question and receive an answer, execute this command:

```bash
python scripts/launch_rag.py --question "Your question here" --config_file "Your configuration file here"
```

Example:

```bash
python scripts/launch_rag.py --question "What says my last medical record ?" --config_file "config/config.yaml"
```

## 📊 Running the Evaluation

To run the evaluation, you need a YAML configuration file specifying all necessary parameters.

Once your configuration file is ready, run the evaluation with:

```bash
python evaluate_pipeline.py config.yaml
```

Or, more simply:

```bash
python evaluate_pipeline.py
```

**Note:** The evaluation dataset is specific to the Chroma/Qdrant database and is located in the persistent directory of the database (as a JSON file).

The script will load the configuration, run the evaluation, and print the results, saving them in `evaluation_results.csv`.

## 🏃‍♂️ Running Parallel Evaluation

To run evaluations in parallel using multiple config files, use the `comparative_evaluation.py` script. This script will search for all YAML config files in the specified directory and run evaluations for each one.

Run the parallel evaluation with:

```bash
python comparative_evaluation.py path_to_config_directory
```

If no directory is specified, the script defaults to `config/comparative_evaluation`.

Evaluation results are printed and logged in the `evaluation.log` file.

Replace `path_to_config_directory` with the path to your directory containing multiple configuration files.

## 📈 Evaluation (Without CLI Command)

The `evaluate_pipeline.py` script offers functions for evaluating the performance of the language model. The `init_evaluation_dataset` function generates a dataset by sampling random chunks from the Chroma database and generating artificial questions. The `launch_eval` function then loads the dataset and performs RAG with each query, tracking successful answers.

The `comparative_evaluation.py` script performs comparative evaluation based on predefined config files. It iterates over each config file, loading the configuration, and launches the evaluation using the `launch_eval` function from `evaluate_pipeline.py`.

## 📄 Document Loading (Without CLI Command)

The `retrieval_utils.py` script provides functions for loading documents into the Chroma database. The `directory_to_vectorstore` function loads documents from a specified directory. The `delete_chroma` function deletes the current Chroma/Qdrant database.
