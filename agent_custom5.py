import asyncio

from browser_use.browser.context import BrowserContextConfig
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

config = BrowserContextConfig(
    # cookies_file="path/to/cookies.json",
    wait_for_network_idle_page_load_time=3.0,
    browser_window_size={'width': 1280, 'height': 1100},
    locale='en-US',
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    viewport_expansion=500,
    save_recording_path='tmp/record_videos'
    # allowed_domains=['google.com', 'wikipedia.org'],
)

# Initialize Browser and Controller
browser = Browser()


class CustomAgent(Agent):
    async def accept_fandom_cookies(self):
        page = await self.browser_context.get_current_page()
        await page.goto("https://starwars.fandom.com/wiki/Obi-Wan_Kenobi")
        # await page.click('[data-tracking-opt-in-accept="true"]')


# Main Execution
async def main():
    async with await browser.new_context(config) as context:
        agent = CustomAgent(
            task="on current page about obi wan check if there are 2 advertisemens visible and loaded. First one is leaderboard ad and second is boxad format and they are most likely iframes, follow each one and tell me where they lead",
            llm=llm,
            browser_context=context,
            use_vision=True
        )
        await agent.accept_fandom_cookies()
        result = await agent.run()
        print(result)


asyncio.run(main())
