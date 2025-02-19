import asyncio

from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from playwright.async_api import BrowserContext

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


class CustomAgent(Agent):
    async def accept_fandom_cookies(self):
        page = await self.browser_context.get_current_page()
        await page.goto("https://starwars.fandom.com/wiki/Obi-Wan_Kenobi")
        await page.click('[data-tracking-opt-in-accept="true"]')


# Main Execution
async def main():
    async with await browser.new_context() as context:
        agent = CustomAgent(
            task="Accept fandom cookies on current page about obi wan and scroll to the very bottom",
            llm=llm,
            browser_context=context
        )
        await agent.accept_fandom_cookies()
        result = await agent.run()
        print(result)


asyncio.run(main())
