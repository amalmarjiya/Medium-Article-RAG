import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

app = FastAPI()

CHUNK_SIZE = 700
OVERLAP_RATIO = 0.15
TOP_K = 7
FETCH_K = 20

EMBED_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"

SYSTEM_PROMPT = """
You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you.

You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context.

If the answer cannot be determined from the provided context, respond exactly:

I don’t know based on the provided Medium articles data.

Do not add any explanation before or after this sentence.

Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

Important formatting rules:
- Follow the user's requested output format exactly.
- If the user asks for only titles, return only titles.
- If the user asks for a title and author, return only the title and author.
- If the user asks for exactly N results, return exactly N results.
- Do not include article IDs or context numbers in the final answer.
- Do not add explanations, reasoning, justification, or commentary unless the user explicitly asks for them.
"""

client = OpenAI(
    api_key=os.environ["LLMOD_API_KEY"],
    base_url=os.environ["LLMOD_BASE_URL"]
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])


class PromptRequest(BaseModel):
    question: str


def clean_title(title):
    return re.sub(r"^\s*\d+\s*:\s*", "", str(title)).strip()


def clean_chunk(chunk):
    chunk = str(chunk)
    chunk = re.sub(r"Title:\s*\d+\s*:\s*", "Title: ", chunk)
    return chunk


def retrieve_distinct_articles(question):
    query_embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=question
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=FETCH_K,
        include_metadata=True
    )

    seen_titles = set()
    matches = []

    for match in results["matches"]:
        md = match["metadata"]
        title = clean_title(md["title"])

        if title not in seen_titles:
            seen_titles.add(title)
            matches.append(match)

        if len(matches) == TOP_K:
            break

    return matches


@app.post("/api/prompt")
def prompt(req: PromptRequest):
    question = req.question

    matches = retrieve_distinct_articles(question)

    context_items = []
    context_text = ""

    for i, match in enumerate(matches):
        md = match["metadata"]

        title = clean_title(md["title"])
        chunk = clean_chunk(md["chunk"])

        context_items.append({
            "article_id": md["article_id"],
            "title": title,
            "chunk": chunk,
            "score": match["score"]
        })

        context_text += f"""
Context {i + 1}
Title: {title}
Authors: {md.get("authors", "")}
Score: {match["score"]}

Chunk:
{chunk}
---
"""

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context_text}

Important:
Follow the user's requested output format exactly.
If the user asks for only titles, return only titles and no explanation.
If the user asks for exactly 3 results, return exactly 3 results.
Do not include article IDs or context numbers in the final answer.
Do not add extra commentary unless the user asks for explanation.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    return {
        "response": response.choices[0].message.content,
        "context": context_items,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt
        }
    }


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }
