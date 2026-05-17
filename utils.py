import os

from dotenv import load_dotenv

from qdrant_client import QdrantClient

from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv(
        "LANGFUSE_PUBLIC_KEY"
    ),

    secret_key=os.getenv(
        "LANGFUSE_SECRET_KEY"
    ),

    host=os.getenv(
        "LANGFUSE_HOST"
    ),
)


def get_qdrant_client():

    return QdrantClient(
        host="localhost",
        port=6333,
        timeout=60,
    )