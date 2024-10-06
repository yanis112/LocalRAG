import argparse
import asyncio
import glob
import os
import time
from datetime import datetime

import aiofiles
import pytz
from dotenv import load_dotenv
from mattermostdriver import Driver
from tqdm.asyncio import tqdm

from src.generation_utils import LLM_answer_v3

# Load the environment variables
load_dotenv()

mm_driver = Driver(
    {
        # from environment variables
        "url": str(os.getenv("URL_MATTERMOST")),
        "token": str(os.getenv("MATTERMOST_TOKEN")),
        "scheme": "https",  # Assuming HTTPS is used
        "port": 443,  # Standard port for HTTPS, adjust if your setup is different
    }
)


def get_username_from_id(user_id):
    """
    Retrieves the username for a given user ID from Mattermost.

    Args:
        user_id (str): The ID of the user.

    Returns:
        str: The username of the user.
    """
    try:
        user_info = mm_driver.users.get_user(user_id)
        return user_info.get("username", "Unknown")
    except Exception as e:
        print(f"Failed to get username for user ID {user_id}: {e}")
        return "Unknown"


def get_list_channel(d):
    """
    Get the list of channels for the user.
    """
    teams_list = mm_driver.teams.get_teams()
    print("Teams list: ", teams_list)
    list_channels = mm_driver.channels.get_list_of_channels_by_ids(
        teams_list[0]["id"]
    )
    return list_channels


def find_messages(d):
    """
    Recursively find all 'message' and 'created_at' values in a dictionary.

    Args:
        d (dict): The dictionary to search for messages and creation dates.

    Returns:
        list: A list of tuples containing the found messages and their creation dates.
    """
    messages = []
    for k, v in d.items():
        if k == "message":
            message = v
            created_at = d.get(
                "create_at", None
            )  # Get the creation date if it exists
            author = d.get("user_id", None)
            # print("AUTHOR:", author)

            author_name = get_username_from_id(author) if author else "Unknown"
            # print("AUTHOR NAME:", author_name)
            messages.append((message, created_at, author_name))
        elif isinstance(v, dict):
            messages.extend(find_messages(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    messages.extend(find_messages(item))
    return messages


def get_all_teams():
    """
    Retrieves all public channels for a team.

    Returns:
        list: A list of dictionaries containing the details of each public channel.
    """
    mm_driver.login()
    teams_list = mm_driver.teams.get_teams()

    # Get the list of public channels for the team
    page = 0
    per_page = 100  # Adjust this value based on how many channels you expect
    public_channels_ids = []
    while True:  # Removed page<4 condition
        try:
            # print("scrapping page: ", page)
            channels_page = mm_driver.teams.get_public_channels(
                teams_list[0]["id"], params={"page": page, "per_page": per_page}
            )
            if not channels_page:  # New condition
                break
            # print("lenght of channels_page:", len(channels_page))
            public_channels_ids.extend(channels_page)
            page += 1
        except Exception:
            break

    # Get the details of each channel
    public_channels = []
    for channel_id in tqdm(
        public_channels_ids, desc="Fetching channel details"
    ):
        try:
            channel = mm_driver.channels.get_channel(channel_id["id"])
            public_channels.append(channel)
        except Exception:
            pass

    # print("Public Channels: ", public_channels)
    return public_channels


def get_list_message(channel_name="Dev"):
    """
    Retrieves a list of messages from the specified channel.
    Args:
        channel_name (str): The name of the channel to retrieve messages from. Defaults to "Dev".
    Returns:
        list: A list of messages, sorted from oldest to most recent. Each message is a tuple containing the message content, the date it was posted, and the author.
    """
    messages = []
    try:
        mm_driver.login()
        print("Logged in successfully!")
    except Exception as e:
        print(f"Failed to log in: {e}")
    # Get the list of public channels for the team
    public_channels = get_all_teams()
    print("All teams obtained successfully!")
    # print("Public Channels: ", public_channels)
    # Find the 'Dev' channel name
    dev_channel_id = None
    for channel in public_channels:
        # print("Channel: ", channel)
        if channel["display_name"] == channel_name:
            dev_channel_id = channel["id"]
            break

    print("Channel ID Optained: ", dev_channel_id)
    # If the 'Dev' channel was found, get its posts
    if dev_channel_id:
        messages = []
        page = 0
        while True:
            try:
                # print("scrapping page: ", page)
                dev_channel_posts = mm_driver.posts.get_posts_for_channel(
                    channel_id=dev_channel_id,
                    params={"page": page, "per_page": 1000, "since": 0},
                )
                message_of_page = find_messages(dev_channel_posts)
                if not message_of_page:
                    break
                messages.extend(message_of_page)
                page += 1
                print("Scrapping page: ", page)
            except Exception:
                break
        original_length = len(messages)
        messages = list(set(messages))  # remove duplicates

        # Remove empty messages and count them
        non_empty_messages = [
            (message, date, author)
            for message, date, author in messages
            if message != ""
        ]
        empty_messages_count = len(messages) - len(non_empty_messages)
        messages = non_empty_messages

        # Sort messages from oldest to most recent
        messages.sort(key=lambda x: x[1] if x[1] is not None else 0)
    else:
        print(f"Channel '{channel_name}' not found")

    return messages


def format_dialogue(messages, channel_name="Dev"):
    """
    Formats a list of messages into blocks of conversation grouped by date.

    Args:
        messages (list): A list of tuples containing the message, its timestamp, and the author.
        channel_name (str, optional): The name of the channel. Defaults to "Dev".

    Returns:
        list: A list of formatted dialogue blocks, each representing a conversation on a specific date.
    """
    # Convert Unix timestamp to date and group messages by date
    messages_by_date = {}
    for message, timestamp, author in messages:
        if timestamp is not None:
            date = datetime.fromtimestamp(timestamp / 1000, pytz.UTC).date()
            if date not in messages_by_date:
                messages_by_date[date] = []
            messages_by_date[date].append((message, author))

    # Format messages into blocks of conversation
    dialogue_blocks = []
    list_titles = []
    for date in sorted(messages_by_date.keys()):
        dialogue = "\n\n".join(
            [
                f"{author}: {message}"
                for message, author in messages_by_date[date]
            ]
        )
        title = f"Dialogue du {date.strftime('%d %B %Y')} sur la plateforme Mattermost dans la channel '{channel_name}'"
        dialogue = "### DIALOGUE ### \n\n" + dialogue
        dialogue_blocks.append(dialogue)
        list_titles.append(title)

    return dialogue_blocks, list_titles


async def save_dialogues(dialogue_blocks, list_titles, path):
    for dialogue, title in tqdm(
        zip(dialogue_blocks, list_titles),
        desc="Saving dialogues",
        total=len(dialogue_blocks),
    ):
        # Replace spaces and special characters with underscores
        filename = (
            title.replace(" ", "_").replace("/", "_").replace(":", "_") + ".md"
        )
        # Create the full file path
        file_path = os.path.join(path, filename)
        # Write the dialogue to the file
        async with aiofiles.open(file_path, "w") as file:
            await file.write(dialogue)


async def process_channel(channel, saving_path="./data/mattermost/"):
    """
    Process the given channel by retrieving messages, formatting dialogues,
    saving dialogues to files, and deleting markdown files with less than 12 words.

    Args:
        channel (str): The name of the channel to process.

    Returns:
        None
    """
    print("Trying to fetch all messages for channel:", channel)
    messages = get_list_message(channel_name=channel)
    print("All messages fetched for channel:", channel)
    dialogue_blocks, list_titles = format_dialogue(
        messages, channel_name=channel
    )
    # we create the directory if it does not exist
    channel_path = os.path.join(saving_path, str(channel))
    if not os.path.exists(channel_path):
        os.makedirs(channel_path)

    await save_dialogues(
        dialogue_blocks=dialogue_blocks,
        list_titles=list_titles,
        path=channel_path,
    )

    # Delete markdown files with less than 12 words and count them
    deleted_count = 0
    async for md_file in tqdm(
        glob.glob(os.path.join(channel_path, "*.md")),
        desc="Deleting short dialogues",
    ):
        async with aiofiles.open(md_file, "r") as file:
            content = await file.read()
            if len(content.split()) < 12:
                os.remove(md_file)
                deleted_count += 1


def dialogue_parser(list_dialogue_blocs):
    """
    Take a readme file as input and produces a summary eliminating the code blocks and other unnecessary information and summarizing the content.

    Args:
        path: str
            The path to the readme file.
    output:
        summary: str
            The summary of the readme file.
    """

    list_summary = []

    for k in tqdm(range(len(list_dialogue_blocs)), desc="Parsing dialogues"):
        prompt = f"Format the following dialogues into a Problem/Solution format. Group these problems/solutions pairs (only if there is a solution) by themes, and keep the dialogues in French. Do not remove any useful information from the discussions except if the questions weren't given any answer. Here are the dialogues: {list_dialogue_blocs[k]}"

        summary = LLM_answer_v3(
            prompt,
            json_formatting=False,
            model_name="llama3-8b-8192",
            stream=False,
            llm_provider="groq",
        )

        list_summary.append(summary)

        print("Summary:", summary)

    return list_summary


async def launch_scrapping(save_path="./data/mattermost_2.0/"):
    """
    Launches the scrapping process for a list of allowed channels.

    This function creates a separate coroutine for each channel in the list of allowed channels and starts the scrapping process for that channel.
    The content of each channel is saved in a separate directory in the 'data/mattermost' folder. The dialogues are then parsed into a day-wise format and saved in markdown files.

    Args:
        None

    Returns:
        None
    """
    list_allowed_channels = [
        "IT @ FR",
        "ENX @ FR",
        "Dev",
        "LLM @ ENX",
        "LowTechForum @ FR",
        "Data Scientists",
    ]

    tasks = []
    for channel in tqdm(
        list_allowed_channels, desc="Starting tasks for channels"
    ):
        task = asyncio.create_task(process_channel(channel, save_path))
        tasks.append(task)

    # Wait for all tasks to finish
    for task in tqdm(
        asyncio.as_completed(tasks),
        desc="Waiting for tasks to finish",
        total=len(tasks),
    ):
        await task


def post_message_to_channel(channel_name, message, channel_id=None):
    """
    Posts a message to a specified channel.

    Args:
        channel_name (str): The name of the channel where the message will be posted.
        message (str): The message to post.
        channel_id (str, optional): The ID of the channel. If None, the channel ID will be looked up by name.

    Returns:
        None
    """
    # Attempt to log in
    try:
        mm_driver.login()
        print("Logged in successfully!")
    except Exception as e:
        print(f"Failed to log in: {e}")
        return

    if channel_id is None:
        # Get the list of public channels for the team and create a mapping of names to IDs
        try:
            public_channels = get_all_teams()
            channel_mapping = {
                channel["display_name"]: channel["id"]
                for channel in public_channels
            }
        except Exception as e:
            print(f"Failed to retrieve channels: {e}")
            return

        # Find the channel ID using the mapping
        channel_id = channel_mapping.get(channel_name)

        print("Corresponding channel ID:", channel_id)

        if not channel_id:
            print(f"Channel '{channel_name}' not found.")
            return

    # Post the message to the channel
    try:
        start_time = time.time()
        mm_driver.posts.create_post(
            options={"channel_id": channel_id, "message": message}
        )
        end_time = time.time()
        print(f"Time taken to post message: {end_time - start_time} seconds.")
        print(f"Message posted successfully to {channel_name}.")
    except Exception as e:
        print(f"Failed to post message to {channel_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch the scrapping process for Mattermost channels."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/mattermost_2.0/",
        help="Path to save the scrapped data",
    )
    args = parser.parse_args()

    asyncio.run(launch_scrapping(save_path=args.save_path))


if __name__ == "__main__":
 
    main()
