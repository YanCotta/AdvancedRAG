"""
Utility functions for Advanced RAG including OpenAI API key retrieval,
index building and query engine creation.
"""

# Standard library imports
import os

# Third-party imports
import numpy as np
import nest_asyncio
from trulens_eval import Feedback, TruLlama, OpenAI
from trulens_eval.feedback import Groundedness
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine

# Apply nested asyncio
nest_asyncio.apply()

# Import settings from config
from AdvancedRAG.config import settings


def get_openai_api_key() -> str:
    """
    Loads the OpenAI API key from environment variables.
    """
    return settings.OPENAI_API_KEY


def get_prebuilt_trulens_recorder(query_engine, app_id: str):
    """
    Prepares a TruLlama recorder with prebuilt feedbacks.
    """
    openai = OpenAI()

    qa_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    qs_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    # Use groundedness without summarize_provider for now
    grounded = Groundedness(groundedness_provider=openai)
    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
    )

    feedbacks = [qa_relevance, qs_relevance, groundedness]
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


def build_sentence_window_index(
    documents, llm, embed_model: str = settings.EMBED_MODEL,
    sentence_window_size: int = settings.SENTENCE_WINDOW_SIZE, save_dir: str = settings.SENTENCE_INDEX_DIR
):
    """
    Builds or loads a sentence window index from documents.
    """
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(documents, service_context=sentence_context)
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )
    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k: int = settings.SIMILARITY_TOP_K, rerank_top_n: int = settings.RERANK_TOP_N
):
    """
    Returns a query engine for the sentence window index with postprocessing.
    """
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=settings.RERANK_MODEL)
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


def build_automerging_index(
    documents, llm, embed_model: str = settings.EMBED_MODEL,
    save_dir: str = settings.MERGING_INDEX_DIR, chunk_sizes=None
):
    """
    Builds or loads an automerging index from documents.
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
        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, service_context=merging_context)
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index, similarity_top_k: int = settings.SIMILARITY_TOP_K, rerank_top_n: int = settings.RERANK_TOP_N
):
    """
    Returns a query engine for the automerging index.
    """
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(base_retriever, automerging_index.storage_context, verbose=True)
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=settings.RERANK_MODEL)
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
