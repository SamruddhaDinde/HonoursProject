from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio
import os

from dotenv import load_dotenv
os.environ["OPENAI_TRACING_DISABLED"] = "true"
load_dotenv()

ollama_client = AsyncOpenAI(
    base_url =os.getenv("OPENAI_BASE_URL","http://localhost:11434/v1"),
    #api_key = os.getenv("OPENAI_API_KEY", "ollama")
)

meta_agent = Agent(
    name="Meta Diagnostician",
    instructions="""You are a senior consultant reviewing assessments from two specialists:
- A clinical text specialist who reasoned from medical knowledge alone
- A radiology specialist who analysed the medical image

Synthesise both assessments and choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <why you chose this answer, considering both specialists, 2-3 sentences>
""",
    model= OpenAIChatCompletionsModel(
        model = "medgemma1.5:4b",
        openai_client=ollama_client
    )
)