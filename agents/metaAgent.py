from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio
import os

ollama_client = AsyncOpenAI(
    base_url =os.getenv("OPENAI_BASE_URL","http://localhost:11434/v1"),
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
)

text_agent = Agent(
    name="Meta Diagnostician",
    instructions = """ You are a senior consultant physician reviewing assessments from two specialist agents.
You will receive two independent analyses of the same medical question.
Your job is to synthesise both opinions and select the final answer.
Respond in this exact format:
ANSWER: <letter only, e.g. A>
REASONING: <why you chose this answer given both inputs>
""",
    model= OpenAIChatCompletionsModel(
        model = "llama3.1:8b",
        openai_client=ollama_client
    )
)