import json
import re
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_description(text):
    if not text or not isinstance(text, str):
        return "No description available"

    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Images: ![alt](url)
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)  # **bold**, *italic*
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)  # __bold__, _italic_
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # # Heading
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)  # `code`
    text = re.sub(r'<[^>]+>', '', text)  # <b>, <i>, etc.
    text = re.sub(r'[ \t]+', ' ', text)  # Multi spaces/tabs
    text = re.sub(r' +\n|\n +', '\n', text)  # Spaces around newlines
    text = text.strip()

    return text or "No description available"

def load_and_create_vector_store(
        json_file=os.path.join("data", "publications.json"),
        persist_dir="chroma_db",
        chunk_size=1000,
        chunk_overlap=200,
        debug=False
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, json_file)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            publications = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        raise ValueError("publications.json is invalid")

    if not isinstance(publications, list):
        raise ValueError(f"Expected a list of entries in {json_path}, found {type(publications)}")

    if debug:
        print(f"Loaded {len(publications)} JSON entries.")

    documents = []
    titles = []
    articles = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for pub in publications:
        pub_id = pub.get('id', 'unknown')
        title = pub.get('title', 'Unknown Title')
        description = pub.get('publication_description', '')

        cleaned_text = clean_description(description)
        if not cleaned_text or cleaned_text == "No description available":
            if debug:
                print(f"Skipping empty article for ID: {pub_id}")
            continue

        titles.append(title)
        articles[pub_id] = cleaned_text

        split_docs = text_splitter.split_text(cleaned_text)
        if debug:
            print(f"Split {pub_id} into {len(split_docs)} chunks")
        for i, chunk in enumerate(split_docs):
            metadata = {
                "id": pub_id,
                "title": title,
                "username": pub.get("username", "unknown"),
                "license": pub.get("license", "unknown"),
                "chunk_index": i
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

    if not documents:
        raise ValueError("No valid documents to process after parsing and splitting.")

    if debug:
        print(f"Parsed and split into {len(documents)} documents.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    if debug:
        print(f"Vector store created with {vectorstore._collection.count()} documents.")

    return vectorstore, titles, articles

if __name__ == "__main__":
    vectorstore, titles, articles = load_and_create_vector_store(debug=True)