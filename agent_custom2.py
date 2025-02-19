import asyncio

from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from browser_use.agent.views import ActionResult

# Define LLM
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.0,
    timeout=100,
)

# Initialize Browser and Controller
browser = Browser(
    config=BrowserConfig(
        headless=False,
    )
)
controller = Controller()


# Define Custom Action
@controller.registry.action("accept fandom cookies")
async def accept_fandom_cookies(url: str):
    print(f"accepting cookies on: {url}")
    context = await browser.new_context()
    page = context.get_current_page()
    await page.click('[data-tracking-opt-in-accept="true"]')
    return ActionResult(extracted_content="Clicked Accept All button")


# Initial Actions
initial_actions = [
    {"open_tab": {"url": "https://starwars.fandom.com/wiki/Obi-Wan_Kenobi"}},
    {"scroll_down": {"amount": 1000}},
]


# Main Execution
async def main():
    agent = Agent(
        task="Accept fandom cookies on current page about obi wan and make sure cookie consent banner is not visible",
        llm=llm,
        browser=browser,
        controller=controller,
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
