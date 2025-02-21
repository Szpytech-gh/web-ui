import asyncio
import os
from playwright.async_api import async_playwright
import pytest
import pytest_asyncio
from browser_use import Agent, Browser, Controller, ActionResult
from langchain_anthropic import ChatAnthropic
from lmnr import Laminar
from pydantic import BaseModel

pytestmark = pytest.mark.asyncio  # Mark all tests in this module as asyncio


class Output(BaseModel):
    page_title: str
    total_cards_top_wiki: str
    visible_cards_top_wiki: int


@pytest_asyncio.fixture
async def browser():
    async with async_playwright() as p:
        browser = Browser()
        yield browser
        await browser.close()


@pytest.fixture
def controller_with_actions():
    """Fixture that provides a controller with all necessary actions registered"""
    controller = Controller(output_model=Output)

    @controller.action('Open website')
    async def open_website(url: str, browser: Browser):
        page = await browser.get_current_page()
        await page.goto(url)
        return ActionResult(extracted_content='Website opened')

    @controller.action('Get page title')
    async def get_page_title(browser: Browser):
        page = await browser.get_current_page()
        title = await page.title()
        return ActionResult(extracted_content=title)

    return controller


@pytest.fixture
def llm():
    return ChatAnthropic(
        model_name="claude-3-5-sonnet-20240620",
        temperature=0.0,
        timeout=100,
    )


@pytest_asyncio.fixture(scope="session")
async def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


async def test_fandom_website(browser, controller_with_actions, llm):
    # Initialize Laminar
    Laminar.initialize(project_api_key=os.getenv('LMNR_PROJECT_API_KEY'))

    # Create an agent with the task
    agent = Agent(
        task="""
        1. Open website https://www.fandom.com/universe/cyberpunk 
        2. give me its title
        3. count total cards in Top Wiki Pages carousel 
        4. count total not visible cards in Top Wiki Pages carousel  
        """,
        llm=llm,
        browser=browser,
        controller=controller_with_actions
    )

    # Run the agent
    history = await agent.run()
    result = history.final_result()

    # Verify the result exists
    assert result is not None, "No result was returned"

    # Parse the result
    parsed = Output.model_validate_json(result)

    assert parsed.page_title == "Everything To Know About Cyberpunk | Fandom"
    assert parsed.total_cards_top_wiki != 0, "Total cards should not be empty"
    assert parsed.visible_cards_top_wiki > 0, "Should have at least one visible card"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
