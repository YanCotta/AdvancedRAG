# AdvancedRAG

## Overview
AdvancedRAG is a project that demonstrates advanced Retrieval Augmented Generation (RAG) pipelines.
It leverages a combination of large language models and vector-based retrieval to answer questions and extract
information from documents with enhanced groundedness and relevance metrics.

## Features
- **Basic RAG Pipeline:** Build a simple vector store index from documents.
- **Advanced RAG Pipelines:** Utilize:
  - Sentence Window Retrieval to extract granular context.
  - Auto-merging Retrieval for hierarchical document processing.
- **Evaluation with TruLens:** Integrate feedback and evaluation modules to assess answer quality.

## Setup
1. **Clone the repository:**
   ```
   git clone https://github.com/YanCotta/AdvancedRAG.git
   ```
2. **Install dependencies:**
   Ensure you have Python 3.8+ installed.
   ```
   pip install -r requirements.txt
   ```
3. **Environment Variables:**
   Create a `.env` file at the root of the project and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

## Usage
- **Basic Pipeline:**
  Run the pipeline script to perform a basic RAG query:
  ```
  python src/pipeline.py
  ```
- **Evaluation Dashboard:**
  The pipeline script automatically launches a local dashboard (http://localhost:8501/) for feedback evaluation using TruLens.

## File Structure
- `/src`: Contains core modules such as:
  - `utils.py`: Utility functions for API key loading, index building, and query engine configuration.
  - `pipeline.py`: The main script demonstrating both basic and advanced RAG pipelines.
- `/data`: Contains input documents and evaluation questions.
- `/LICENSE`: MIT License details.

## Educational Insights
- The code is structured with educational inline comments to help you understand each processing step.
- The utility functions are modularized to easily extend or modify the RAG components.
- Evaluation modules integrated with TruLens help you assess output relevance and groundedness.

## Contributing
Contributions and improvements are welcome. Please fork the repository and create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.