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
    instructions = """ You are an experienced radiologist.
    You will be given a medical image and a clinical question about it.
    Analyze the question carefully and Answer the question.
    Respond in this exact format:
    Answer: <YES or No>
    Reasoning: <what you observed in the image that led to this answer in 2-3 sentences>
""",
    model= OpenAIChatCompletionsModel(
        model = "llava:7b",
        openai_client=ollama_client
    )
)