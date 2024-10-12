from github import Github
from dotenv import load_dotenv
import os

load_dotenv()

# Authentication with your personal access token
g = Github(os.getenv("GITHUB_TOKEN"))

# Access the repository
repo = g.get_repo("yanis112/High-Quality-Promps-for-Imagen3")

# Get the content of the README.md file
file = repo.get_contents("README.md")
current_content = file.decoded_content.decode()

# Modify the content
new_content = current_content + "\n\nThis is a new line added by the Python script."

# Update the file on GitHub
repo.update_file(
    path="README.md",
    message="Update README via Python script",
    content=new_content,
    sha=file.sha
)

print("The README.md file has been successfully updated.")