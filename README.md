# AdvancedRAG

## Overview
A sophisticated implementation of advanced Retrieval Augmented Generation (RAG) techniques, featuring multi-strategy retrieval, automated evaluation, and modular architecture.

## Key Features
- **Multi-Strategy Retrieval Pipeline**
  - AutoMerging Retrieval with hierarchical node parsing
  - Sentence Window Retrieval for granular context
  - Cross-encoder reranking for enhanced relevance
  - Multi-hop reasoning capabilities

- **Advanced Evaluation Framework**
  - Integrated TruLens evaluation
  - Confidence scoring and analysis
  - Automated groundedness assessment
  - Performance metrics dashboard

## File Structure
- **/AutoMergingRetrieval**
  Core implementation of auto-merging techniques:
  - `utils.py`: Core utilities for document processing and indexing
  - `AutoMergingRetrieval.py`: Implementation of hierarchical node merging
  
- **/AdvancedRAGPipeline**  
  Complete RAG pipeline implementation:
  - **/src**:  
    - `utils.py`: Pipeline utilities and configuration
    - `pipeline.py`: Orchestrates RAG workflows with evaluation
  - **/data**:  
    Sample data and evaluation sets

- **/data**  
  Holds shared resources like PDF documents and evaluation question files.

## Techniques and Methodologies
- **AutoMerging Retrieval:**  
  Utilizes hierarchical node parsing to merge document nodes across varying levels of granularity, resulting in more contextualized retrieval.
  
- **Sentence Window Retrieval:**  
  Extracts text in overlapping windows to capture granular context, enhancing retrieval precision.

- **TruLens Evaluation:**  
  Integrates feedback mechanisms that measure answer relevance and groundedness, ensuring high-quality responses.

## Implementation Details
- **AutoMergingRetrieval:**
  - Implements dynamic node size adjustment
  - Uses similarity-based merging strategies
  - Supports customizable merging thresholds

- **Advanced RAG Pipeline:**
  - Integrates multiple retrieval strategies
  - Features automated evaluation loops
  - Provides detailed performance metrics

## Setup & Usage
1. **Clone the repository:**
   ```
   git clone https://github.com/YanCotta/AdvancedRAG.git
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Environment Variables:**  
   Create a `.env` file at the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```
4. **Run the pipelines:**
   - For basic and auto-merging retrieval, execute the scripts in `/src`.
   - For the full RAG pipeline with additional evaluations, run the scripts in `/AdvancedRAGPipeline/src`.

## Contributing & License
Contributions are welcome. See the MIT License in the LICENSE file for details.