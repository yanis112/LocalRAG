

import json
import os
import random
import string

#go two directories up using pathlib
from pathlib import Path
import sys
import matplotlib.pyplot as plt


# Get the directory of the current file
current_dir = Path(__file__).parent

# Go two directories up
up_path = current_dir.parent.parent

print("UP PATH:",up_path)

# Add the path to sys.path
sys.path.append(str(up_path))


def parse_versions(data):
    document_text = ""
    metadata = {}
    assets = []

    versions = data.get('versions', {}).get('edges', [])
    if not versions:
        return document_text, metadata, assets

    for version in versions:
        node = version.get('node', {})
        translated_versions = node.get('translatedVersions', [])
        for translated_version in translated_versions:
            body = translated_version.get('body', {})
            if body:
                state = body.get('state', {})
                document = state.get('document', {})
                nodes = document.get('nodes', [])
                for node in nodes:
                    node_type = node.get('type')
                    if node_type == 'html':
                        document_text += node['data']['html'] + "\n"
                    elif node_type == 'paragraph':
                        text_content = ''.join(n.get('text', '') for n in node.get('nodes', []))
                        document_text += text_content + "\n"
                    elif node_type == 'heading':
                        heading_content = ''.join(n.get('text', '') for n in node.get('nodes', []))
                        document_text += heading_content + "\n"
                    elif node_type == 'link':
                        link_content = ''.join(n.get('text', '') for n in node.get('nodes', []))
                        document_text += link_content + "\n"
            
            # Extracting metadata
            metadata['title'] = translated_version.get('title', 'No title available')
            author = translated_version.get('author', {})
            if isinstance(author, dict):
                metadata['author'] = author.get('id', 'Unknown author')
            else:
                metadata['author'] = 'Unknown author'
            metadata['created_at'] = translated_version.get('created', 'No creation date available')
            metadata['published_at'] = translated_version.get('publishedAt', 'No publication date available')
            metadata['updated_at'] = node.get('updated', 'No update date available')

            # Extracting assets
            assets.extend(translated_version.get('assets', []))

    return document_text, metadata, assets


def extract_and_save_stories(json_path, save_path):
    """
    This function extracts stories from a JSON file and saves them as individual text files.
    It also generates a histogram of word counts in the stories.

    Args:
    json_path (str): The path to the JSON file.
    save_path (str): The directory where the text files will be saved.

    Returns:
    None
    """
    # Load the JSON file
    with open(json_path) as f:
        data = []
        for line in f:
            # Ignore empty lines
            if line.strip():
                data.append(json.loads(line))

    # Create a list to store the word counts
    word_counts = []

    # Create sets to store the titles and contents
    titles = set()
    contents = set()

    # Create a counter for duplicates
    duplicates = 0

    # Iterate over the data
    k=0
    for item in data:
        # Parse the item to get the document text and metadata
        document_text, metadata, assets = parse_versions(item)
        if document_text != "":
            # Concatenate the document text and metadata
            content = f"{document_text}\n{metadata}\n{str(assets)}"

            #delete all duplicated spaces and newlines
            content = ' '.join(content.split())

            #length of the content
            length_content = len(content.split())

            # Skip if the content or title is a duplicate
            title = metadata.get('title', 'default') 
            if title in titles or content in contents:
                duplicates += 1
                continue

            titles.add(title)
            contents.add(content)

            # Add the word count to the list
            word_counts.append(len(content.split()))

            if title is None: #we give it a random title
                title = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

            # Get the title from the metadata to use as the filename
            filename = title[0:30].replace('/','_') + '.txt'
            k+=1

            if length_content>1000000:
                print("HUGEEE TEXT:",filename)

            if length_content<1000000:
                #add the path of the save dir to the filename
                filename=os.path.join(save_path,filename)

                # Write the content to a text file
                with open(filename, 'w') as f:
                    f.write(content)

    print("NUMBER OF FILES CREATED:",k)
    print("NUMBER OF DUPLICATES:", duplicates)

    print("MAX WORD COUNT:",max(word_counts))
    print("MIN WORD COUNT:",min(word_counts))
    print("AVERAGE WORD COUNT:",sum(word_counts) / len(word_counts))

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=30)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of Word Counts')
    plt.savefig(os.path.join(save_path, "histogram_word_counts.png"))

if __name__ == "__main__":
    # Open the file data/knowledge_attic/knowledge_stories.json
    json_path = "data/knowledge_attic/knowledge_stories.json"
    # save path
    save_path = "data/knowledge_attic/"

    #delete all the txt files in the current directory
    import glob
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Delete all .txt files in the directory
    for f in glob.glob(os.path.join(save_path, "*.txt")):
        os.remove(f)

    print("Extracting stories from:", json_path)

    # Extract and save the stories
    extract_and_save_stories(json_path, save_path)