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

text_agent = Agent(
    name="Clinical Text Analyst",
    instructions = """ You are an experienced clinical physician.
You will be given a medical question without any images.
Reason purely from clinical knowledge to answer.
Your answer should be concise and direct — it could be yes/no, 
a body part, a direction, a size, or any other short clinical finding.
Respond in this exact format:
ANSWER: <your concise answer>
REASONING: <your clinical reasoning in 2-3 sentences>
""",
    model= OpenAIChatCompletionsModel(
        model = "llama3.1:8b",
        openai_client=ollama_client
    )
)