"""
Entry point for the Medical Decision Support System.

Run from the simple_multiagent_1 directory:
    python main.py

Requires in your .env file:
    GROQ_API_KEY=your_key_here
    TAVILY_API_KEY=your_key_here

Try these example queries to test each agent:
    Symptom agent:  "I have a persistent cough, fever, and night sweats for 3 weeks"
    Research agent: "What are the current first-line treatments for hypertension?"
    Both agents:    "I have chest pain when breathing — what could cause this and 
                     what does research say about treatment?"
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph.workflow import build_graph

load_dotenv()


def validate_env():
    """Check required API keys are present before building the graph."""
    missing = [key for key in ["GROQ_API_KEY", "TAVILY_API_KEY"]
               if not os.getenv(key)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Add them to your .env file."
        )


def run():
    validate_env()

    print("\n" + "="*60)
    print("   Medical Decision Support System — Prototype v0.1")
    print("="*60)
    print("⚠️  FOR EDUCATIONAL PURPOSES ONLY.")
    print("    Always consult a qualified healthcare professional.")
    print("="*60)
    print("\nType your question below. Type 'exit' to quit.\n")

    # Build the graph once — compilation happens here, not on each query
    graph = build_graph()
    print("[System] Graph compiled successfully. Ready.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[System] Exiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("[System] Goodbye.")
            break

        # Wrap user input in a HumanMessage and invoke the graph.
        # Each call starts fresh — this prototype has no cross-session memory.
        # To add memory: compile with checkpointer=MemorySaver() and pass
        # config={"configurable": {"thread_id": "user_123"}} to invoke().
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "agent_calls": 0,
            "last_agent": ""
        }

        try:
            result = graph.invoke(initial_state)
            # Find the last message with non-empty text content (skip tool messages).
            final_response = next(
                (m.content for m in reversed(result["messages"])
                 if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip()),
                "[No response generated]"
            )
            print(f"\nAssistant:\n{final_response}\n")

        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}")
            print("Check your API keys and try again.\n")


if __name__ == "__main__":
    run()
