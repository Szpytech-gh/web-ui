import asyncio
import os

from browser_use import Agent, Browser, Controller, ActionResult
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from lmnr import Laminar
from pydantic import BaseModel, SecretStr

# Create a browser instance
config = BrowserContextConfig(
    # cookies_file="path/to/cookies.json",
    # wait_for_network_idle_page_load_time=3.0,
    browser_window_size={'width': 1300, 'height': 900},
    locale='en-US',
    # user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    # allowed_domains=['google.com', 'wikipedia.org'],
)
browser = Browser()
context = BrowserContext(browser=browser, config=config)

api_key = os.getenv("DEEPSEEK_API_KEY")
# Initialize the OpenAI LLM (you'll need to set up your API key)
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.0,
    timeout=100,
)
planner_llm = ChatOpenAI(base_url='https://api.deepseek.com/v1', model='deepseek-chat', api_key=SecretStr(api_key))


class Output(BaseModel):
    page_title: str
    total_cards_top_wiki: str
    visible_cards_top_wiki: int


# Create a controller and define a custom action
controller = Controller(output_model=Output)

Laminar.initialize(project_api_key=os.getenv('LMNR_PROJECT_API_KEY'))


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


@controller.action('count Total Top Wiki Pages carousel cards')
async def count_total_carousel_cards(browser: Browser):
    page = await browser.get_current_page()

    # CSS selector targeting the Top Wiki Pages carousel cards
    selector = '.wds-widget-frame:has(h2.wds-widget-frame__title[title="Top Wiki Pages"]) .wds-card-link'

    card_count = await page.locator(selector).count()
    return ActionResult(extracted_content=str(card_count))


async def main():
    # Create an agent with a specific task
    agent = Agent(
        task="""
        1. Open website https://www.fandom.com/universe/cyberpunk 
        2. give me its title
        3. count total cards in Top Wiki Pages carousel 
        4. using vision, count total visible cards in Top Wiki Pages carousel  
        """,
        llm=llm,
        planner_llm=planner_llm,
        browser_context=context,
        controller=controller,
        use_vision=True
    )

    # Run the agent
    history = await agent.run()

    print(history)

    result = history.final_result()
    if result:
        parsed = Output.model_validate_json(result)

        print('\n--------------------------------')
        print(f'Title:            {parsed.page_title}')
        print(f'Total cards:              {parsed.total_cards_top_wiki}')
        print(f'Visible cards:         {parsed.visible_cards_top_wiki}')
    else:
        print('No result')

    # Don't forget to close the browser when you're done
    await context.close()


asyncio.run(main())
