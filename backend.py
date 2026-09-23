"""
Resume Chat

Answers questions about a resume that has already been indexed into
Pinecone by a separate ingestion process.
Combines vector search and LLM generation.

Setup:
    pip install fastapi uvicorn anthropic pinecone python-dotenv voyageai

    .env file:
        ANTHROPIC_API_KEY=your-key-here
        PINECONE_API_KEY=your-key-here
        PINECONE_INDEX_NAME=your-index-here
        PDF_NAME=your-filename-here
"""

import os
import logging
from datetime import date
import voyageai
from anthropic import Anthropic
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

anthropic_client = Anthropic()
voyage_client = voyageai.Client()
pinecone_client = Pinecone()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PDF_NAME = os.getenv("PDF_NAME")
EMBEDDING_MODEL = "voyage-3"
GENERATION_MODEL = "claude-sonnet-5"
EMBEDDING_DIMENSIONS = 1024
# MIN_SIMILARITY = 0.7

def get_index():
    return pinecone_client.Index(PINECONE_INDEX_NAME)

def embed_text(text: str) -> list[float]:
    response = voyage_client.embed(
        [text],
        model="voyage-3",
        input_type="query"
    )
    return response.embeddings[0]

def ask_question(question: str, index: object):
    """Asks a question about the resume."""
    logger.info(f"Question for {PDF_NAME}: {question[:80]}...")

    query_embedding = embed_text(question)

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter={"pdf_name": {"$eq": PDF_NAME}}
    )

    chunks = [
        match.metadata.get("text", "")
        for match in results.matches
        # if match.score > MIN_SIMILARITY
    ]

    logger.info(f"Retrieved {len(chunks)} relevant chunks")

    if not chunks:
        def no_context():
            return "I could not find relevant sections in the resume to answer your question. Try rephrasing or download the resume."

    content = "\n\n---\n\n".join(chunks)

    response = anthropic_client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=2048, # Caps the response length. Set this intentionally based on what your use case actually needs. Leaving it too high burns unnecessary tokens and cost. Leaving it too low truncates responses mid-sentence.
            output_config={"effort": "high"},
            system=f"""You are a document assistant.
                Answer questions based only on the provided document sections.
                Be direct and specific.
                If the answer is not clearly in the provided sections, say so.
                Do not fabricate information.
                When referencing specific information, indicate which part of the document it came from.
                Today's date is {date.today().isoformat()}. Use it for any date arithmetic,
                such as computing how long a "Present"-ended role has lasted.
                """,
            messages=[{
                "role": "user",
                "content": f"Document sections:\n\n{content}\n\nQuestion: {question}"
            }]
        )
    texts = [block.text for block in response.content if block.type == 'text']
    return texts[0]