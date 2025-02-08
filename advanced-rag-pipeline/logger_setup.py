import logging

def configure_logger(log_file="app.log"):
    logger = logging.getLogger("AdvancedRAG")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

def log_chunking_error(logger, file_name, chunk_identifier, error):
    logger.error(f"Chunking failed for file {file_name}, chunk {chunk_identifier}: {error}")

def log_embedding_error(logger, file_name, chunk_identifier, error):
    logger.error(f"Embedding failed for file {file_name}, chunk {chunk_identifier}: {error}")

# Additional logging/error handling utilities can be added here as needed.
