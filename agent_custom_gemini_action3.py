import asyncio
import os

from browser_use import Agent, Browser, Controller, ActionResult
from lmnr import Laminar
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI

# Create a controller and define a custom action
controller = Controller()
api_key = os.getenv("GEMINI_API_KEY")
Laminar.initialize(project_api_key=os.getenv('LMNR_PROJECT_API_KEY'))


@controller.action('Open website')
async def open_website(url: str, browser: Browser):
    page = await browser.get_current_page()
    await page.goto(url)
    return ActionResult(extracted_content='Website opened')

# Create a browser instance
browser = Browser()

# Initialize the OpenAI LLM (you'll need to set up your API key)
# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(os.getenv('GEMINI_API_KEY')))


async def main():
    # Create an agent with a specific task
    agent = Agent(
        task="""
        1. Open website https://www.fandom.com/universe/cyberpunk 
        2. give me its title
        3. count total cards in Top Wiki Pages carousel 
        4. count total not visible cards in Top Wiki Pages carousel  
        """,
        llm=llm,
        browser=browser,
        controller=controller
    )

    # Run the agent
    result = await agent.run()

    print(result)

    # Don't forget to close the browser when you're done
    await browser.close()


asyncio.run(main())
