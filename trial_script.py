from data.loader import load_nejm, split_context_and_question

dataset = load_nejm(n_samples=20, seed=42)
for example in dataset:
    context, question = split_context_and_question(example["question"])
    print(f"\n--- Case {example['image_id']} ---")
    print(f"CONTEXT: {context}")
    print(f"QUESTION: {question}")