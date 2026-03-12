from brain.brain import ResearchBrain

def main():
    brain = ResearchBrain()

    # This should LOAD from storage (not re-ingest)
    brain.ingest("sample.pdf")

    print("\n📄 Chat with Research Paper (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break

        answer = brain.ask(question)
        print("\nAI:", answer, "\n")


if __name__ == "__main__":
    main()
