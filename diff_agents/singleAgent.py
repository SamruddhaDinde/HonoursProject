from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv

os.environ["OPENAI_TRACING_DISABLED"] = "true"
load_dotenv()

ollama_client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)

single_agent = Agent(
    name="Medical Diagnostician",
    instructions="""You are an experienced physician with expertise in clinical reasoning and image interpretation.
You will be given a clinical case, multiple-choice options, and a medical image.
Integrate the clinical history and the visual findings to choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <how the clinical history and image findings together support your choice, 2-3 sentences>
""",
    model=OpenAIChatCompletionsModel(
        model="qwen2.5vl:7b",   
        openai_client=ollama_client
    )
)