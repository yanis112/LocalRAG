from github import Github
from dotenv import load_dotenv
import os
import re
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()

# Authentication with your personal access token
g = Github(os.getenv("GITHUB_TOKEN"))

# Access the repository
repo = g.get_repo("yanis112/SOTA_machine_learning")

# Get the content of the README.md file
file = repo.get_contents("README.md")
current_content = file.decoded_content.decode()

class LongDocEditor:
    def __init__(self, content):
        self.content = content
        self.model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
        self.dataset_embeddings = None
        self.section_embeddings = None

    def get_sections(self):
        """
        Extract sections from the document content.

        This method finds all sections in the document content that start with '###'.
        Each section is identified by a title line starting with '###' and includes
        the content until the next '###' or the end of the document.

        Returns:
            list of dict: A list of dictionaries, each containing:
            - 'title' (str): The title of the section.
            - 'content' (str): The content of the section.
            - 'start_line' (int): The starting line number of the section.
            - 'end_line' (int): The ending line number of the section.
        """
        sections = []
        # Find all sections starting with '###'
        section_pattern = re.compile(r'(### .+?)(?=\n### |\Z)', re.DOTALL)
        matches = section_pattern.finditer(self.content)
        lines = self.content.split('\n')
        
        for match in matches:
            title_match = re.match(r'(### .+)', match.group(1))
            if title_match:
                title = title_match.group(1).strip()
                body = match.group(1)[len(title):].strip()
                start_pos = match.start(1)
                end_pos = match.end(1)
                start_line = self.content.count('\n', 0, start_pos) + 1
                end_line = self.content.count('\n', 0, end_pos)
                sections.append({
                    'title': title,
                    'content': body,
                    'start_line': start_line,
                    'end_line': end_line
                })
        return sections

    def get_section_titles(self):
        """
        Extracts and returns a list of section titles from the document content.

        A section title is defined as a line starting with '###'.

        Returns:
            list: A list of section titles found in the document content.
        """
        titles = []
        # Find all section titles starting with '###'
        section_pattern = re.compile(r'(### .+)', re.DOTALL)
        matches = section_pattern.findall(self.content)
        for match in matches:
            titles.append(match.strip())
        return titles

    def compute_embeddings(self):
        """
        Compute embeddings for the document sections.

        This method performs a two-stage embedding process:
        1. Embeds a subset of the corpus (minicorpus) using the model.
        2. Embeds all sections using the previously computed embeddings of the minicorpus.

        The embeddings are stored in the instance variables `dataset_embeddings` and `section_embeddings`.

        Returns:
            None
        """
        sections = self.get_sections()
        docs = [section['title']+' \n'+section['content'] for section in sections]
        # First stage: embed a subset of the corpus
        minicorpus_size = self.model[0].config.transductive_corpus_size
        minicorpus_docs = docs[:minicorpus_size]
        self.dataset_embeddings = self.model.encode(
            minicorpus_docs,
            prompt_name="document",
            convert_to_tensor=True
        )
        # Second stage: embed the sections
        self.section_embeddings = self.model.encode(
            docs,
            prompt_name="document",
            dataset_embeddings=self.dataset_embeddings,
            convert_to_tensor=True
        )

    def most_relevant_section(self, input_text):
        """
        Identifies the most relevant section of the document based on the input text.

        This method computes the embeddings for the input text and compares them with
        precomputed section embeddings to find the section with the highest similarity.

        Args:
            input_text (str): The text input for which the most relevant section is to be found.

        Returns:
            str: The most relevant section of the document.
        """
        if self.section_embeddings is None:
            self.compute_embeddings()
        # Embed the input text
        query_embedding = self.model.encode(
            [input_text],
            prompt_name="query",
            dataset_embeddings=self.dataset_embeddings,
            convert_to_tensor=True
        )
        # Compute similarity
        similarities = self.model.similarity(query_embedding, self.section_embeddings)
        print("## Similarities:", similarities)
        topk_values, topk_indices = similarities.topk(1)
        most_relevant_index = topk_indices[0].item()
        sections = self.get_sections()
        return sections[most_relevant_index]

if __name__ == "__main__":
    editor = LongDocEditor(current_content)
    sections = editor.get_sections()
    print("Number of sections:", len(sections))

    if len(sections) > 1:
        print("Second section:", sections[1])

    # Compute embeddings for sections
    editor.compute_embeddings()

    # Find the most relevant section for a given input text
    input_text = "What are the latest advancements in image generation?"
    most_relevant = editor.most_relevant_section(input_text)
    print("Most relevant section for the input text:")
    print(most_relevant)