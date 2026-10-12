import json
import os
import pprint
import sys
from pathlib import Path

import chromadb
import pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sentence_transformers import SentenceTransformer


# incarcam fisierul -> chunk-uri (spargem bucati) -> embeddings -> incarcam in baza de date vectoriala -> semantic search pe DB vectorial -> returnam rezultatele.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"
FILE_PATH = "StarMaker.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Question about the contents of StarMaker.pdf.")


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

    collection.upsert(
        ids=ids,
        # this is where embedding happens for each chunk
        # normalize_embeddings normalizes all vectors, their values are between [0, 1]
        embeddings=model.encode(documents, normalize_embeddings=True).tolist(),
        documents=documents,
        # aici inseram datele vectoriale in baza de date:
        metadatas=metadatas,
    )


def complete(client: OpenAI, messages: list[dict], tools: list[dict]):
    return client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto",
    ).choices[0].message


def answer_with_context(client: OpenAI, collection, model, args: str) -> str:
    try:
        request = SearchRequest.model_validate(json.loads(args))
        vector = model.encode(request.question, normalize_embeddings=True).tolist()
        result = collection.query(
            query_embeddings=[vector],
            n_results=8,
            include=["documents", "metadatas"],
        )
        context_pieces = []
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        # construim un string structurat pentru LLM
        for i in range(len(documents)):
            document = documents[i]
            metadata = metadatas[i]
            context_pieces.append(
                f"{document}\nSource: {metadata['source']}, page {metadata['page']}"
            )

        context = "\n\n".join(context_pieces)
        return client.chat.completions.create(
            model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "Answer only from the supplied context. If it does not answer the question, say so. Cite every factual claim with the provided source.",
                },
                {"role": "user", "content": f"Question: {request.question}\n\nContext:\n{context}"},
            ],
        ).choices[0].message.content
    except (json.JSONDecodeError, ValidationError, KeyError) as error:
        return f"error: {error}"


def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    index_file(collection, model, Path(FILE_PATH))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_starmaker",
                "description": "Searches StarMaker.pdf and answers from it. Use only when the user asks about StarMaker.pdf or its contents.",
                "parameters": SearchRequest.model_json_schema(),
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use search_starmaker only for questions about StarMaker.pdf or its contents. Use it before answering those questions. Answer other questions normally.",
        }
    ]

    while True:
        question = input("you> ")
        if question == "quit":
            break

        messages.append({"role": "user", "content": question})
        response = complete(client, messages, tools)
        messages.append(response.model_dump(exclude_none=True))

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "search_starmaker":
                    output = answer_with_context(client, collection, model, tool_call.function.arguments)
                else:
                    output = f"error: unknown tool {tool_call.function.name}"
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

            response = complete(client, messages, tools)
            messages.append(response.model_dump(exclude_none=True))

        print("agent>", response.content)


if __name__ == "__main__":
    main()
