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

vision_agent = Agent(
    name="Radiology Vision Analyst",
    instructions="""You are an experienced radiologist.
You will be given a medical image alongside a clinical case and multiple-choice options.
Examine the image carefully and choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <what you observed in the image and how it supports your choice, 2-3 sentences>
""",
    model= OpenAIChatCompletionsModel(
        model = "medgemma1.5:4b",
        openai_client=ollama_client
    )
)