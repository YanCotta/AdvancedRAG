#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------
# Advanced RAG Pipeline: Auto-merging Retrieval Module
# Demonstrates multi-layer chunking and index merging for advanced RAG.
# ------------------------------------------------------------------------------

import os
import openai
from llama_index import (
    Document,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from AdvancedRAG.config import settings
from fastapi import FastAPI, Query
from typing import List

# Initialize OpenAI API key.
openai.api_key = settings.OPENAI_API_KEY

# ------------------------------------------------------------------------------
# Configuration Parameters
# Adjust these parameters to optimize the merging process.
# ------------------------------------------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
NODE_LIMIT = 10

# ------------------------------------------------------------------------------
# Step 1: Define Node Parsers
# Configures hierarchical parsing for document chunking.
# ------------------------------------------------------------------------------
def get_node_parser():
    """
    Initializes and returns a HierarchicalNodeParser configured for chunking.

    Returns:
        HierarchicalNodeParser: Configured node parser.
    """
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        node_parser_ids=[
            "sentence",
            "paragraph",
        ],
    )
    return node_parser

# ------------------------------------------------------------------------------
# Step 2: Build Auto-Merging Index
# Creates an auto-merging index from the document set.
# ------------------------------------------------------------------------------
def build_automerging_index(
    documents, llm, embed_model=settings.EMBED_MODEL, save_dir="merging_index"
):
    """
    Builds an auto-merging index from a list of documents.

    Args:
        documents (list): List of Document objects.
        llm: Language model instance.
        embed_model (str): Embedding model name. Defaults to settings.EMBED_MODEL.
        save_dir (str): Directory to save the index. Defaults to "merging_index".

    Returns:
        VectorStoreIndex: The constructed vector store index.
    """
    node_parser = get_node_parser()
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    index = VectorStoreIndex(
        leaf_nodes, service_context=ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    )
    index.storage_context.persist(persist_dir=save_dir)
    return index

# ------------------------------------------------------------------------------
# Step 3: Load Auto-Merging Index
# Loads an existing auto-merging index from storage.
# ------------------------------------------------------------------------------
def load_automerging_index(
    llm, embed_model=settings.EMBED_MODEL, save_dir="merging_index"
):
    """
    Loads an auto-merging index from a specified directory.

    Args:
        llm: Language model instance.
        embed_model (str): Embedding model name. Defaults to settings.EMBED_MODEL.
        save_dir (str): Directory where the index is stored. Defaults to "merging_index".

    Returns:
        VectorStoreIndex: The loaded vector store index.
    """
    storage_context = StorageContext.from_defaults(persist_dir=save_dir)
    index = load_index_from_storage(
        storage_context,
        service_context=ServiceContext.from_defaults(llm=llm, embed_model=embed_model),
    )
    return index

# ------------------------------------------------------------------------------
# Step 4: Get Auto-Merging Query Engine
# Retrieves a query engine configured for auto-merging.
# ------------------------------------------------------------------------------
def get_automerging_query_engine(automerging_index):
    """
    Retrieves a query engine for the auto-merging index.

    Args:
        automerging_index (VectorStoreIndex): The auto-merging index.

    Returns:
        QueryEngine: A query engine configured for the index.
    """
    query_engine = automerging_index.as_query_engine(
        similarity_top_k=6,
        # NOTE: set higher chunk size
        node_postprocessors=[
            utils.MetadataReplacementPostProcessor(target_metadata_key="window"),
            utils.AutoMergingRetriever(
                automerging_index.storage_context,
                similarity_top_k=6,
                rerank_top_n=2,
            ),
        ],
    )
    return query_engine

# ------------------------------------------------------------------------------
# Step 5: Automatic Summaries or Context Windows
# Generates smaller summarized chunks for extremely large documents.
# ------------------------------------------------------------------------------
def generate_summary(text, llm):
    """
    Generates a summary for a given text using a language model.

    Args:
        text (str): The text to summarize.
        llm: Language model instance.

    Returns:
        str: The summarized text.
    """
    prompt = f"Please provide a concise summary of the following text:\n{text}\nSummary:"
    response = llm.complete(prompt)
    return str(response)

def build_automerging_index_with_summaries(
    documents, llm, embed_model=settings.EMBED_MODEL, save_dir="merging_index_with_summaries"
):
    """
    Builds an auto-merging index with automatic summaries for large documents.

    Args:
        documents (list): List of Document objects.
        llm: Language model instance.
        embed_model (str): Embedding model name. Defaults to settings.EMBED_MODEL.
        save_dir (str): Directory to save the index.

    Returns:
        VectorStoreIndex: The constructed vector store index.
    """
    node_parser = get_node_parser()
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    # Generate summaries for large nodes
    for node in leaf_nodes:
        if len(node.text) > CHUNK_SIZE * 2:  # Define "large" as twice the chunk size
            node.text = generate_summary(node.text, llm)

    index = VectorStoreIndex(
        leaf_nodes, service_context=ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    )
    index.storage_context.persist(persist_dir=save_dir)
    return index

# ------------------------------------------------------------------------------
# Step 6: Fallback Mechanism
# Implements a fallback mechanism using keyword search if semantic embedding confidence is low.
# ------------------------------------------------------------------------------
def hybrid_query(query_text, query_engine, keyword_query_engine, confidence_threshold=0.6):
    """
    Executes a hybrid query using semantic embedding and keyword search.

    Args:
        query_text (str): The query text.
        query_engine: The semantic embedding query engine.
        keyword_query_engine: The keyword search query engine.
        confidence_threshold (float): Confidence threshold to switch to keyword search.

    Returns:
        str: The response from either the semantic embedding or keyword search engine.
    """
    response = query_engine.query(query_text)
    confidence_score = response.metadata.get("confidence_score", 0.0)  # Assuming your engine provides a confidence score

    if confidence_score < confidence_threshold:
        print("Low confidence in semantic search, falling back to keyword search.")
        keyword_response = keyword_query_engine.query(query_text)
        return keyword_response
    else:
        return response

# Dummy keyword query engine for demonstration purposes
class DummyKeywordQueryEngine:
    def query(self, query_text):
        return f"Keyword search result for: {query_text}"

# ------------------------------------------------------------------------------
# Step 7: Endpoint or API Server Support
# Exposes an API using FastAPI to query the advanced retrieval from a web interface or other service.
# ------------------------------------------------------------------------------
app = FastAPI()

# Load the automerging index at startup
try:
    automerging_index = load_automerging_index(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1))
    query_engine = get_automerging_query_engine(automerging_index)
    print("Automerging index loaded successfully.")
except Exception as e:
    automerging_index = None
    query_engine = None
    print(f"Error loading automerging index: {e}")

@app.get("/query/")
async def query_index(q: str = Query(..., title="Query text")):
    """
    Endpoint to query the auto-merging index.

    Args:
        q (str): The query text.

    Returns:
        dict: The query response.
    """
    if query_engine is None:
        return {"error": "Query engine not initialized. Check server logs for loading errors."}

    try:
        response = query_engine.query(q)
        return {"response": str(response)}
    except Exception as e:
        return {"error": str(e)}

# Example usage within the script (for testing purposes)
if __name__ == "__main__":
    # Load data (replace with your data loading logic)
    from llama_index import SimpleDirectoryReader
    documents = SimpleDirectoryReader(
        input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
    ).load_data()

    # Consolidate documents
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    # Build index with summaries
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    automerging_index_with_summaries = build_automerging_index_with_summaries(
        [document], llm, save_dir="merging_index_with_summaries"
    )
    query_engine_with_summaries = get_automerging_query_engine(automerging_index_with_summaries)

    # Example query
    query_text = "How do I build a portfolio of AI projects?"
    response = query_engine_with_summaries.query(query_text)
    print(f"Query: {query_text}\nResponse: {response}\n")

    # Example usage of hybrid query
    keyword_query_engine = DummyKeywordQueryEngine()
    hybrid_response = hybrid_query(query_text, query_engine_with_summaries, keyword_query_engine)
    print(f"Hybrid Query Response: {hybrid_response}\n")

    # To run the FastAPI server, use: uvicorn your_script_name:app --reload
    # (e.g., uvicorn AutoMergingRetrieval:app --reload)
    # Then, access the query endpoint in your browser or using a tool like curl:
    # http://127.0.0.1:8000/query/?q=your_query_here
