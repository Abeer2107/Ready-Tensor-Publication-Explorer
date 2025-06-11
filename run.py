from src.cli import run_cli
from src.document_loader import load_and_create_vector_store

def main():
    vectorstore = load_and_create_vector_store()
    run_cli(vectorstore)

if __name__ == "__main__":
    main()