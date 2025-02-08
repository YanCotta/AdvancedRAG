#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------
# Advanced RAG Pipeline: This script demonstrates both basic and advanced
# retrieval augmented generation (RAG) workflows.
# Educational comments are included to explain each step of the pipeline.
# ------------------------------------------------------------------------------

"""
This module orchestrates the RAG workflows, integrating Sentence Window Retrieval and automated index building,
along with TruLens evaluation to assess output groundedness and relevance. It focuses on end-to-end pipeline
execution and evaluation.
"""

from AdvancedRAG import utils
import os
import openai

# Initialize OpenAI API key using our utility function.
# openai.api_key = utils.get_openai_api_key()
from AdvancedRAG.config import settings
openai.api_key = settings.OPENAI_API_KEY

# Add these imports at the top
from .advanced_retrieval import AdvancedRetrieval
from .confidence_summary import ConfidenceSummary

# ------------------------------------------------------------------------------
# Step 1: Load source documents.
# Here we load a PDF document that serves as a knowledge base.
# ------------------------------------------------------------------------------
from llama_index import SimpleDirectoryReader
def load_data(input_files):
    """
    Loads data from the specified files using SimpleDirectoryReader.

    Args:
        input_files (list): A list of file paths to load.

    Returns:
        list: A list of Document objects loaded from the files.
    """
    documents = SimpleDirectoryReader(
        input_files=input_files
    ).load_data()
    return documents

documents = load_data(["./eBook-How-to-Build-a-Career-in-AI.pdf"])

print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])

# ------------------------------------------------------------------------------
# Step 2: Prepare a single Document from multiple parts.
# This aggregates the document contents for indexing.
# ------------------------------------------------------------------------------
from llama_index import Document
def consolidate_documents(documents):
    """
    Consolidates a list of Document objects into a single Document.

    Args:
        documents (list): A list of Document objects.

    Returns:
        Document: A single Document object containing the combined text from all input documents.
    """
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    return document

document = consolidate_documents(documents)

# ------------------------------------------------------------------------------
# Step 3: Basic RAG pipeline using a vector store index.
# Sets up the service context with a chosen LLM and embedding model.
# ------------------------------------------------------------------------------
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=settings.EMBED_MODEL
)
index = VectorStoreIndex.from_documents([document], service_context=service_context)

# Modify the query engine creation to include advanced features
def create_advanced_query_engine(index):
    """Creates an advanced query engine with additional retrieval techniques."""
    advanced_retrieval = AdvancedRetrieval()
    confidence_summary = ConfidenceSummary()
    
    base_engine = index.as_query_engine()
    
    def advanced_query(query_str: str):
        # Get initial results
        initial_nodes = index.as_retriever().retrieve(query_str)
        
        # Apply cross-encoder reranking
        reranked_nodes = advanced_retrieval.rerank_nodes(query_str, initial_nodes)
        
        # Perform multi-hop reasoning if needed
        answer, confidence = advanced_retrieval.multi_hop_query(
            query_str, reranked_nodes, llm
        )
        
        # Analyze confidence and generate summary
        analysis = confidence_summary.analyze_response(answer, confidence)
        return analysis
    
    return advanced_query

# Replace the existing query engine creation with:
query_engine = create_advanced_query_engine(index)

# Run a sample query.
response = query_engine("What are steps to take when finding projects to build your experience?")
print(str(response))

# ------------------------------------------------------------------------------
# Step 4: Set up evaluation using TruLens.
# Loads evaluation questions from file and prepares the TruLens recorder.
# ------------------------------------------------------------------------------
def load_eval_questions(file_path):
    """
    Loads evaluation questions from a specified file.

    Args:
        file_path (str): The path to the file containing evaluation questions.

    Returns:
        list: A list of evaluation questions.
    """
    eval_questions = []
    with open(file_path, 'r') as file:
        for line in file:
            item = line.strip()
            print(item)
            eval_questions.append(item)
    return eval_questions

eval_questions = load_eval_questions('eval_questions.txt')

new_question = "What is the right AI job for me?"
eval_questions.append(new_question)
print(eval_questions)

from trulens_eval import Tru
tru = Tru()
tru.reset_database()

# Use prebuilt recorder for evaluating the query engine.
from AdvancedRAG.src.utils import get_prebuilt_trulens_recorder
tru_recorder = get_prebuilt_trulens_recorder(query_engine, app_id="Direct Query Engine")
with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
print(records.head())

# Launch the evaluation dashboard on localhost.
tru.run_dashboard()

# ------------------------------------------------------------------------------
# Step 5: Advanced RAG pipeline - Sentence Window Retrieval.
# Constructs a specialized index that uses sentence windows.
# ------------------------------------------------------------------------------
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
sentence_index = utils.build_sentence_window_index(
    document,
    llm,
    embed_model=settings.EMBED_MODEL,
    save_dir="sentence_index"
)

sentence_window_engine = utils.get_sentence_window_query_engine(sentence_index)
window_response = sentence_window_engine.query("how do I get started on a personal project in AI?")
print(str(window_response))

tru.reset_database()
tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id="Sentence Window Query Engine"
)
for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))

tru.get_leaderboard(app_ids=[])
tru.run_dashboard()

# ------------------------------------------------------------------------------
# Step 6: Advanced RAG pipeline - Auto-merging Retrieval.
# This index automatically merges nodes across different granularities.
# ------------------------------------------------------------------------------
automerging_index = utils.build_automerging_index(
    documents,
    llm,
    embed_model=settings.EMBED_MODEL,
    save_dir="merging_index"
)

automerging_query_engine = utils.get_automerging_query_engine(automerging_index)
auto_merging_response = automerging_query_engine.query("How do I build a portfolio of AI projects?")
print(str(auto_merging_response))

tru.reset_database()
tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")
for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)

tru.get_leaderboard(app_ids=[])
tru.run_dashboard()

