from langchain_anthropic import ChatAnthropic
from browser_use import Agent
from dotenv import load_dotenv

load_dotenv()

import asyncio

llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.0,
    timeout=100,  # Increase for complex tasks
)


async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=llm,
    )
    result = await agent.run()
    print(result)


asyncio.run(main())

