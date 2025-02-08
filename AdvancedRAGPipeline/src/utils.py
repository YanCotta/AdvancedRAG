#!pip install python-dotenv

import os
import numpy as np
import nest_asyncio
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)
from trulens_eval.feedback import Groundedness
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine

nest_asyncio.apply()

# Import settings from config
from AdvancedRAG.config import settings

# ------------------------------------------------------------------------------
# Utility functions for environment API key loading and feedback configuration.
# These functions help abstract configuration and index building for our RAG pipeline.
# ------------------------------------------------------------------------------

def get_openai_api_key():
    """
    Loads the OpenAI API key from the environment using a .env file.
    """
    return settings.OPENAI_API_KEY


def get_hf_api_key():
    """
    Loads the HuggingFace API key from the environment using a .env file.
    """
    return settings.HUGGINGFACE_API_KEY

openai = OpenAI()

# Create feedback objects with detailed in-line explanations.
qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()  # Measures relevance based on both inputs and outputs.
)

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()  # Only consider the question.
    .on(TruLlama.select_source_nodes().node.text)  # Then on the text of source nodes.
    .aggregate(np.mean)  # Takes the mean as an aggregate statistic.
)

# Use groundedness to verify that answers are well-supported.
grounded = Groundedness(groundedness_provider=openai)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)  # Aggregate method for grounded statements.
)

feedbacks = [qa_relevance, qs_relevance, groundedness]

def get_trulens_recorder(query_engine, feedbacks, app_id):
    """
    Creates a TruLlama recorder with a specific query engine, feedback, and app_id.
    """
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

def get_prebuilt_trulens_recorder(query_engine, app_id):
    """
    A convenience function to create a prebuilt TruLlama recorder using default feedbacks.
    """
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    return tru_recorder

# ------------------------------------------------------------------------------
# Functions related to index creation and query engine setup for retrieval tasks.
# ------------------------------------------------------------------------------

def build_sentence_window_index(
    document, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index"
):
    """
    Builds or loads a sentence window index for a given document.
    - Splits document into sentence windows.
    - Persists the index if not already existing.
    """
    # Create a SentenceWindowNodeParser with default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=settings.EMBED_MODEL,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    """
    Sets up the query engine for sentence window retrieval.
    - Uses postprocessors to replace metadata and rerank the results.
    """
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model=settings.RERANK_MODEL
    )
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    """
    Builds or loads an auto-merging index from a list of documents.
    - Splits the document into hierarchical nodes for various granularities.
    """
    chunk_sizes = chunk_sizes or settings.CHUNK_SIZES
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    """
    Configures and returns a query engine for the auto-merging index.
    - Wraps a retriever with an AutoMergingRetriever and adds a re-ranker.
    """
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model=settings.RERANK_MODEL
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
