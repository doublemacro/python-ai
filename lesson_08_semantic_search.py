import sys
from pathlib import Path

import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer

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


def main():
    file_path = Path(FILE_PATH)
    mrk = pymupdf4llm.to_markdown(str(file_path))
    chunks = chunk_text(mrk)

    model = SentenceTransformer(MODEL_NAME)
    # this is where embedding happens for each chunk
    # normalize_embeddings normalizes all vectors, their values are between [0, 1]
    vectors = model.encode(chunks, normalize_embeddings=True).tolist()
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    collection.delete(where={"source": file_path.name})
    # aici inseram datele vectoriale in baza de date:
    collection.upsert(
        ids = [f"{file_path.name}:{index}" for index in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"source": file_path.name} for i in chunks]
    )
    # query:
    q = input("you> ")
    query_vector = model.encode(q, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=3, include=["documents", "distances"])

    print(results)



if __name__ == "__main__":
    main()










