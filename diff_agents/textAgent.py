from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio
import os
from dotenv import load_dotenv
os.environ["OPENAI_TRACING_DISABLED"] = "true"

load_dotenv()

ollama_client = AsyncOpenAI(
    base_url =os.getenv("OPENAI_BASE_URL","http://localhost:11434/v1"),
    #api_key = os.getenv("OPENAI_API_KEY", "ollama")
    timeout=120.0,
)

text_agent = Agent(
    name="Clinical Text Analyst",
    instructions="""You are an experienced clinical physician.
You will be given a clinical case description and multiple-choice options, without any image.
Reason from clinical knowledge to choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <your clinical reasoning in 2-3 sentences>
""",
    model= OpenAIChatCompletionsModel(
        model = "medgemma1.5:4b",
        openai_client=ollama_client
    )
)