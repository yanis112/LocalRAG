
from browser_use import Agent
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

async def main():
    agent = Agent(
        task="Find who is Pierre Yves Rougeyron.",
    #     llm=ChatGoogleGenerativeAI(
    #         model='gemini-2.0-flash-exp')
    # )
        llm=ChatGroq(
            model='llama-3.3-70b-versatile'))
    result = await agent.run()
    print(result)

asyncio.run(main())

