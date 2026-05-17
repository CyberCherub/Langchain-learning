import os
import nest_asyncio

from unstructured.partition.auto import partition

nest_asyncio.apply()

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
)

from llama_index.core.settings import Settings

from llama_index.vector_stores.qdrant import (
    QdrantVectorStore
)

from llama_index.core.postprocessor import (
    SentenceTransformerRerank
)

from llama_index.llms.groq import Groq

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

from utils import (
    get_qdrant_client,
    langfuse,
)

# =========================
# SETTINGS
# =========================

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

Settings.embed_model = (
    HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
)

# =========================
# CLIENTS
# =========================

qdrant_client = (
    get_qdrant_client()
)

# =========================
# VECTOR STORE
# =========================

vector_store = (
    QdrantVectorStore(
        client=qdrant_client,
        collection_name="enterprise_rag",
    )
)

storage_context = (
    StorageContext.from_defaults(
        vector_store=vector_store
    )
)

# =========================
# PARSER
# =========================

def parse_file(path):

    elements = partition(
        filename=path
    )

    return "\n".join(
        [str(el) for el in elements]
    )

# =========================
# INGESTION
# =========================

def ingest_documents(
    data_dir="data"
):

    documents = []

    supported_files = (
        ".pdf",
        ".docx",
        ".pptx",
        ".txt",
        ".md",
    )

    for file in os.listdir(
        data_dir
    ):

        if file.lower().endswith(
            supported_files
        ):

            path = os.path.join(
                data_dir,
                file
            )

            try:

                text = parse_file(
                    path
                )

                if text.strip():

                    documents.append(
                        Document(
                            text=text,
                            metadata={
                                "source": file
                            }
                        )
                    )

                    print(
                        f"Ingested: {file}"
                    )

            except Exception as e:

                print(
                    f"Failed {file}: {e}"
                )

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

# =========================
# LOAD INDEX
# =========================

def load_index():

    return (
        VectorStoreIndex
        .from_vector_store(
            vector_store=vector_store
        )
    )

# =========================
# QUERY
# =========================

def query_rag(query):

    index = load_index()

    reranker = (
        SentenceTransformerRerank(
            model=(
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            ),
            top_n=3,
        )
    )

    query_engine = (
        index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[
                reranker
            ],
        )
    )

    response = query_engine.query(
        query
    )

    sources = []

    for node in (
        response.source_nodes
    ):

        sources.append(
            node.metadata
        )

    # Langfuse event logging
    langfuse.create_event(
        name="rag_query",
        input={
            "query": query
        },
        output={
            "response": str(response)
        }
    )

    langfuse.flush()

    return (
        str(response),
        sources
    )