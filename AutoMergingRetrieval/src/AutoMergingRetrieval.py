"""
Main script for demonstrating Auto-merging Retrieval with AdvancedRAG.
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Import utilities and required components
import AdvancedRAG.AutoMergingRetrieval.src.utils as utils
from llama_index import SimpleDirectoryReader, Document, StorageContext, load_index_from_storage, ServiceContext, VectorStoreIndex
from llama_index.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms import OpenAI
from trulens_eval import Tru

# Import settings from config
from AdvancedRAG.config import settings

def load_documents() -> list:
    # Load documents from the given PDF
    return SimpleDirectoryReader(input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]).load_data()

def display_document_info(documents: list):
    # Display information about loaded documents
    print(type(documents))
    print(len(documents))
    print(type(documents[0]))
    print(documents[0])

def build_and_persist_index(documents, context: ServiceContext, nodes, save_dir: str) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    if not os.path.exists(save_dir):
        index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=context)
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir), service_context=context)
    return index

def run_evaluations(eval_questions: list, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

def main():
    documents = load_documents()
    display_document_info(documents)

    # Consolidate documents into one instance
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    # Build hierarchical node parser and nodes
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
    nodes = node_parser.get_nodes_from_documents([document])
    leaf_nodes = get_leaf_nodes(nodes)
    print(leaf_nodes[30].text)
    nodes_by_id = {node.node_id: node for node in nodes}
    parent_node = nodes_by_id[leaf_nodes[30].parent_node.node_id]
    print(parent_node.text)

    # Create LLM and ServiceContext for auto-merging retrieval
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    auto_merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=settings.EMBED_MODEL,
        node_parser=node_parser,
    )

    # Build or load automerging index (example using local folder "./merging_index")
    automerging_index = build_and_persist_index(leaf_nodes, auto_merging_context, leaf_nodes, "./merging_index")
    
    # Optionally, build a fresh index with different chunk sizes for layered tests
    auto_merging_index_0 = utils.build_automerging_index(
        documents, llm=llm, embed_model=settings.EMBED_MODEL, save_dir="merging_index_0", chunk_sizes=[2048, 512]
    )
    auto_merging_engine_0 = utils.get_automerging_query_engine(auto_merging_index_0, similarity_top_k=12, rerank_top_n=6)

    # TruLens evaluation setup: reset database and run evaluations
    Tru().reset_database()
    tru_recorder = utils.get_prebuilt_trulens_recorder(auto_merging_engine_0, app_id='app_0')

    # Load evaluation questions from file
    eval_questions = []
    with open('generated_questions.text', 'r') as file:
        for line in file:
            eval_questions.append(line.strip())
    
    run_evaluations(eval_questions, tru_recorder, auto_merging_engine_0)

    # Additional layers using three-layer index
    auto_merging_index_1 = utils.build_automerging_index(
        documents, llm=llm, embed_model=settings.EMBED_MODEL, save_dir="merging_index_1", chunk_sizes=[2048, 512, 128]
    )
    auto_merging_engine_1 = utils.get_automerging_query_engine(auto_merging_index_1, similarity_top_k=12, rerank_top_n=6)
    tru_recorder = utils.get_prebuilt_trulens_recorder(auto_merging_engine_1, app_id='app_1')
    run_evaluations(eval_questions, tru_recorder, auto_merging_engine_1)

    # Display TruLens results and launch dashboard
    Tru().get_leaderboard(app_ids=[])
    Tru().run_dashboard()

if __name__ == "__main__":
    main()

