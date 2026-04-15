from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio
import os

ollama_client = AsyncOpenAI(
    base_url =os.getenv("OPENAI_BASE_URL","http://localhost:11434/v1"),
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
)

text_agent = Agent(
    name="Clinical Text Analyst",
    instructions = """ You are an experienced clinical physician.
    You will be given a medical question with multiple choice options.
    Analyze the question carefully and select the best answer.
    Respond in this exact format:
    Answer: <leeter only, e.g. A>
    Reasoning: <your clinical reasoning in 2-3 sentences>
""",
    model= OpenAIChatCompletionsModel(
        model = "llama3.1:8b",
        openai_client=ollama_client
    )
)