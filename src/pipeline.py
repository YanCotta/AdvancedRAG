#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------
# Advanced RAG Pipeline: This script demonstrates both basic and advanced
# retrieval augmented generation (RAG) workflows.
# Educational comments are included to explain each step of the pipeline.
# ------------------------------------------------------------------------------

import AdvancedRAG.src.utils as utils
import os
import openai

# Initialize OpenAI API key using our utility function.
openai.api_key = utils.get_openai_api_key()

# ------------------------------------------------------------------------------
# Step 1: Load source documents.
# Here we load a PDF document that serves as a knowledge base.
# ------------------------------------------------------------------------------
from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])

# ------------------------------------------------------------------------------
# Step 2: Prepare a single Document from multiple parts.
# This aggregates the document contents for indexing.
# ------------------------------------------------------------------------------
from llama_index import Document
document = Document(text="\n\n".join([doc.text for doc in documents]))

# ------------------------------------------------------------------------------
# Step 3: Basic RAG pipeline using a vector store index.
# Sets up the service context with a chosen LLM and embedding model.
# ------------------------------------------------------------------------------
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document], service_context=service_context)

# Create a query engine for basic queries.
query_engine = index.as_query_engine()

# Run a sample query.
response = query_engine.query("What are steps to take when finding projects to build your experience?")
print(str(response))

# ------------------------------------------------------------------------------
# Step 4: Set up evaluation using TruLens.
# Loads evaluation questions from file and prepares the TruLens recorder.
# ------------------------------------------------------------------------------
eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        item = line.strip()
        print(item)
        eval_questions.append(item)

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
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
print(records.head())

# Launch the evaluation dashboard on localhost.
tru.run_dashboard()

# ------------------------------------------------------------------------------
# Step 5: Advanced RAG pipeline - Sentence Window Retrieval.
# Constructs a specialized index that uses sentence windows.
# ------------------------------------------------------------------------------
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
from AdvancedRAG.src.utils import build_sentence_window_index
sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)

from AdvancedRAG.src.utils import get_sentence_window_query_engine
sentence_window_engine = get_sentence_window_query_engine(sentence_index)
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
from AdvancedRAG.src.utils import build_automerging_index
automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)

from AdvancedRAG.src.utils import get_automerging_query_engine
automerging_query_engine = get_automerging_query_engine(automerging_index)
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

