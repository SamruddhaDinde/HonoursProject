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
    instructions = """ ou are an experienced radiologist.
You will be given a medical image and a clinical question about it.
Examine the image carefully and answer the question.
Your answer should be concise and direct — it could be yes/no,
a body part, a direction, a size, or any other short visual finding.
Respond in this exact format:
ANSWER: <your concise answer>
REASONING: <what you observed in the image in 2-3 sentences>
""",
    model= OpenAIChatCompletionsModel(
        model = "llava:7b",
        openai_client=ollama_client
    )
)