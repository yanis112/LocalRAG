import chainlit as cl
import yaml
from src.chainlit_app_utils import load_config,process_query


@cl.step(type="tool")
def tool():
    # Simulate a running task
    cl.sleep(2)

    return "Response from the tool!"

def load_config():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/idea.svg",
            ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="/public/learn.svg",
            ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="/public/terminal.svg",
            ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="/public/write.svg",
            )
        ]


@cl.on_message
async def main(message: cl.Message):
    # Call the tool
    res=process_query(message.content,load_config())

    # Send the final answer.
    await cl.Message(content=res).send()