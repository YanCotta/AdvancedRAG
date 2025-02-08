from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict
import torch
from llama_index.schema import NodeWithScore, TextNode
import numpy as np

class AdvancedRetrieval:
    def __init__(self):
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def rerank_nodes(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Reranks retrieved nodes using cross-encoder."""
        if not nodes:
            return nodes
            
        texts = [node.node.text for node in nodes]
        pairs = [[query, text] for text in texts]
        scores = self.cross_encoder.predict(pairs)
        
        # Sort nodes by cross-encoder scores
        scored_nodes = list(zip(nodes, scores))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in scored_nodes]

    def multi_hop_query(self, query: str, nodes: List[NodeWithScore], 
                       llm, max_hops: int = 2) -> Tuple[str, float]:
        """Performs multi-hop reasoning across documents."""
        context = ""
        confidence = 1.0
        
        for hop in range(max_hops):
            # Generate hop-specific query
            if hop == 0:
                hop_query = query
            else:
                hop_query = llm.predict(
                    f"Based on what we know so far: {context}\n"
                    f"What additional information do we need to answer: {query}"
                )
            
            # Get and rerank nodes for this hop
            hop_nodes = self.rerank_nodes(hop_query, nodes)
            if not hop_nodes:
                break
                
            # Add most relevant context
            context += "\n" + hop_nodes[0].node.text
            confidence *= float(hop_nodes[0].score)
        
        # Generate final answer
        answer = llm.predict(
            f"Based on this context: {context}\n"
            f"Please answer: {query}"
        )
        
        return answer, confidence
