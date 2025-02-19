import asyncio

from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig

# Define LLM
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.0,
    timeout=100,
)

initial_actions = [
    {"open_tab": {"url": "https://starwars.fandom.com/wiki/Obi-Wan_Kenobi"}},
    {"scroll_down": {"amount": 1000}},
    {"click_element": {"xpath": "//div[@data-tracking-opt-in-accept=\"true\"]"}},
]

agent = Agent(
    task="Accept fandom cookies on current page about obi wan and scroll to the bottom",
    llm=llm,
)

# Main Execution
async def main():
    result = await agent.run()
    print(result)


asyncio.run(main())