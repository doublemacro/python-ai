import os
import sys
from pathlib import Path

import pprint
import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


# incarcam fisierul -> chunk-uri (spargem bucati) -> embeddings -> incarcam in baza de date vectoriala -> semantic search pe DB vectorial -> returnam rezultatele.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"
FILE_PATH = "StarMaker.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        # get a slice of text, using start as the starting index.
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        # make a step, update the index(start)
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def index_file(collection, model, file_path):
    pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    records = []

    for page in pages:
        page_number = page["metadata"]["page_number"]
        page_text = page["text"]

        chunks = chunk_text(page_text)
        chunk_number = 0

        for chunk in chunks:
            # tuples -> ("john", "carpenter", 33)
            #                 0            1   2
            records.append((chunk, page_number, chunk_number))
            chunk_number += 1

    collection.delete(where={"source": file_path.name})
    ids = []
    documents = []
    metadatas = []

    # records = [("text here", 1, 1), ("some more text", 1, 2), ...]
    for record in records:
        chunk = record[0]
        page_number = record[1]
        chunk_number = record[2]
        ids.append(f"{file_path.name}:p{page_number}:c{chunk_number}")
        documents.append(chunk)
        metadatas.append(
            {
                "source": file_path.name,
                "page": page_number,
                "chunk": chunk_number,
            }
        )

    # this is where embedding happens for each chunk
    # normalize_embeddings normalizes all vectors, their values are between [0, 1]
    vectors = model.encode(documents, normalize_embeddings=True).tolist()
    # aici inseram datele vectoriale in baza de date:
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
    )


def main():
    load_dotenv()

    clientOpenAI = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY")
    )

    file_path = Path(FILE_PATH)
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    index_file(collection, model, file_path)
    # query:
    while True:
        q = input("you> ")
        query_vector = model.encode(q, normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=[query_vector], n_results=8, include=["documents", "distances"])

        response = answer_with_context(clientOpenAI, q, result)
        pprint.pp(response.content)


def answer_with_context(client: OpenAI, question, result):
    context_pieces = []
    documents = result["documents"][0]

    # construim un string structurat pentru LLM
    for i in range(len(documents)):
        doc = documents[i]
        context_pieces.append(f"{doc}")

    context = "\n\n".join(context_pieces)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that finds answers given a user question, and some context. Answer only from the supplied context, if it answers the question. If it doesn't answer the question, say so. Also, cite every factual claim, with its source."
            },
            {
                "role": "user",
                "content": f"User asks: {question}\n\nContext: {context}"
            }
        ],
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
    ).choices[0].message
    return response



if __name__ == "__main__":
    main()








