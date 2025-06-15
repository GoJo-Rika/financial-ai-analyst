import os

from agno.agent import Agent
from agno.models.groq import Groq
from agno.playground import Playground
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["AGNO_API_KEY"] = os.getenv("AGNO_API_KEY")

## web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[DuckDuckGoTools()],
    instructions=["Alway include sources"],
    show_tool_calls=True,
    markdown=True,
)

## Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        ),
    ],
    instructions=["Alwyas use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

playground = Playground(
    agents=[finance_agent, web_search_agent],
    name="Basic Agent",
    description="A playground for basic agent",
    app_id="basic-agent",
)

app = playground.get_app()

if __name__ == "__main__":
    playground.serve("playground:app", reload=True)
