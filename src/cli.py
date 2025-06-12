from src.document_loader import load_and_create_vector_store
from src.rag_pipeline import RAGPipeline

def run_cli(vectorstore):
    pipeline = RAGPipeline()
    print("Welcome to ReadyTensor Publication Explorer!")
    print("Ask questions about ReadyTensor publications (type 'exit' to quit)")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            print("Exiting ReadyTensor Publication Explorer. Goodbye!")
            break

        try:
            # Create chain with a retriever for all documents
            chain = pipeline.create_chain(vectorstore.as_retriever(search_kwargs={"k": 1}))
            response = chain.invoke({"query": question})
            # Handle RetrievalQA output (string or dict with 'result')
            response_text = response if isinstance(response, str) else response.get("result", str(response))
            print(f"\nAnswer: {response_text}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main():
    try:
        vectorstore, _, _ = load_and_create_vector_store()
        run_cli(vectorstore)
    except Exception as e:
        print(f"Failed to initialize: {str(e)}")

if __name__ == "__main__":
    main()