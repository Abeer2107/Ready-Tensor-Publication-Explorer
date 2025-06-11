from src.rag_pipeline import create_rag_chain

def run_cli(vectorstore):
    chain = create_rag_chain(vectorstore)
    print("Welcome to Ready Tensor Publication Explorer!")
    print("Ask questions about Ready Tensor publications")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            print("Exiting Ready Tensor Publication Explorer. Goodbye!")
            break

        try:
            response = chain.invoke({"query": question})["result"]
            print(f"\nAnswer: {response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")