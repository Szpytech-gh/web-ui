import os
import sys
from pathlib import Path

from browser_use.agent.views import ActionResult
from langchain_anthropic import ChatAnthropic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

import pyperclip

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

browser = Browser(
    config=BrowserConfig(
        headless=False,
    )
)
controller = Controller()


@controller.registry.action('Copy text to clipboard')
def copy_to_clipboard(text: str):
    pyperclip.copy(text)
    return ActionResult(extracted_content=text)


@controller.registry.action('Paste text from clipboard')
async def paste_from_clipboard(browser: BrowserContext):
    text = pyperclip.paste()
    # send text to browser
    page = await browser.get_current_page()
    await page.keyboard.type(text)

    return ActionResult(extracted_content=text)


async def main():
    task = f'Copy the text "Hello, world!" to the clipboard, then go to google.com and paste the text'
    llm = ChatAnthropic(
        model_name="claude-3-5-sonnet-20240620",
        temperature=0.0,
        timeout=100,
    )
    agent = Agent(
        task=task,
        llm=llm,
        controller=controller,
        browser=browser,
    )

    await agent.run()
    await browser.close()

    input('Press Enter to close...')


if __name__ == '__main__':
    asyncio.run(main())