import asyncio
from browser_use import Agent, Browser, Controller, ActionResult
from langchain_anthropic import ChatAnthropic

# Create a controller and define a custom action
controller = Controller()


@controller.action('Open website')
async def open_website(url: str, browser: Browser):
    page = await browser.get_current_page()
    await page.goto(url)
    return ActionResult(extracted_content='Website opened')

# Create a browser instance
browser = Browser()

# Initialize the OpenAI LLM (you'll need to set up your API key)
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.0,
    timeout=100,
)


async def main():
    # Create an agent with a specific task
    agent = Agent(
        task="Open website https://www.fandom.com/universe/cyberpunk and give me its title.",
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
