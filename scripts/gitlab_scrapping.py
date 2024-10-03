import asyncio
import os
import argparse

import aiohttp
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

from src.generation_utils import LLM_answer_v3

# Load environment variables
load_dotenv()

# Get GitLab URL and private token from environment variables
GITLAB_URL = os.getenv("GITLAB_URL")
PRIVATE_TOKEN = os.getenv("PRIVATE_TOKEN")


async def get_project_readme(session, project):
    """Fetch the README file of a given project."""
    try:
        async with session.get(
            f"{GITLAB_URL}/api/v4/projects/{project['id']}/repository/files/README.md/raw",
            params={"ref": "master"},
            headers={"PRIVATE-TOKEN": PRIVATE_TOKEN},
        ) as response:
            if response.status == 200:
                readme = await response.text()
                return project["name"], readme
            else:
                return project["name"], None
    except Exception as e:
        return project["name"], str(e)


async def save_readme(readme, project_name, directory):
    """Save the README file of a given project."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f"{directory}/{project_name}.md", "w") as f:
        f.write(readme)


async def get_all_projects(session):
    """Fetch all projects using GitLab API with pagination."""
    projects = []
    page = 1
    print("GITLAB_URL", GITLAB_URL)
    while True:
        async with session.get(
            f"{GITLAB_URL}/api/v4/projects",
            params={
                "page": page,
                "per_page": 1000,
                "include_hidden": "true",
                "include_pending_delete": "true",
            },
            headers={"PRIVATE-TOKEN": PRIVATE_TOKEN},
        ) as response:
            if response.status == 200:
                page_projects = await response.json()
                if not page_projects:
                    break
                projects.extend(page_projects)
                page += 1
            else:
                break
    return projects


async def get_list_readmes(directory):
    """Main function to fetch all projects and their README files."""
    readmes = []
    async with aiohttp.ClientSession() as session:
        projects = await get_all_projects(session)
        print(f"Total repositories found: {len(projects)}")
        tasks = [get_project_readme(session, project) for project in projects]
        for f in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Processing projects",
        ):
            project_name, readme = await f
            if readme:
                readmes.append(readme)
                await save_readme(readme, project_name, directory)
        print(f"\nTotal repositories processed: {len(readmes)}")
    return readmes


def readme_parser(path):
    """
    Take a readme file as input and producess a summary eliminating the code blocks and other unnecessary information and summarizing the content.

    Args:
        path: str
            The path to the readme file.
    output:
        summary: str
            The summary of the readme file.
    """
    import yaml
    
    #open the config file config/config.yaml
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    #find in the config the LLM provider and the model name
    llm_provider = config["llm_provider"]
    model_name = config["model_name"]

    # open the readme file in markdown format
    with open(path, "r") as file:
        data = file.read()

    # get the name of the project from the path
    project_name = path.split("/")[-1].split(".")[0]

    # define a project title to be used in the prompt
    project_title = f"#### PROJECT NAME: {project_name}"

    # remove all unnecessary spaces
    data = data.strip()

    # add the title at the beginning of the readme content
    data = project_title + "\n" + data

    class Summary(BaseModel):
        summary: str

    prompt = f"Summarize the content of the following readme file of a project to explain the goal of the project, the technologies used (algorithms, models, tools, etc.), the main features of the project, and the domain of application of the project. Remove the code, commands, parameters and project structure considerations, precise the jargon and acronyms . Here is the content of the readme file to process, answer the summary without preamble: <readme>{data}</readme>."

    json_summary = LLM_answer_v3(
        prompt,
        json_formatting=False,
        pydantic_object=Summary,
        model_name=model_name,
        llm_provider=llm_provider,
        stream=False,
    )

    return json_summary


def run_readme_parser(folder_path):
    """
    List all files in the given directory, apply the readme_parser function to each file, and overwrite the content of the readme files with the summary generated by readme_parser.

    Args:
        folder_path: str
            The path to the directory containing the readme files.
    """
    # List all files in the directory
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Apply readme_parser to each file and overwrite the content with the summary
    for file in tqdm(files, desc="Summarizing/parsing files"):
        summary = readme_parser(file)
        with open(file, "w") as f:
            f.write(summary)


def main():
    parser = argparse.ArgumentParser(description="GitLab README Fetcher and Parser")
    parser.add_argument("--allow_parsing", type=bool, default=False, help="Allow parsing of README files")
    parser.add_argument("--path", type=str, default="data/gitlab", help="Path to save README files")
    args = parser.parse_args()

    readmes = asyncio.run(get_list_readmes(args.path))

    print("Total readme files fetched: ", len(readmes))
    if args.allow_parsing:
        print("Processing readme files to generate summaries...")
        run_readme_parser(args.path)


if __name__ == "__main__":
    main()