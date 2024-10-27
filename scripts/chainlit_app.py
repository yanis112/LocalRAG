import chainlit as cl
from src.chainlit_app_utils import process_query, load_config


async def add_goodby(text):
    return text + " Goodbye!"

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the configuration from the user session
    config = load_config()
    
    print("config: ", config)
    print("message: ", message.content)

    # Process the user's query and generate an answer
    answer = await process_query(message.content, config)
    print("Answer: ", answer)

    # Stream the answer back to the user
    await cl.Message(content=answer).send()