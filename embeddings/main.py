import time
import logging
from sentence_transformers import SentenceTransformer
import torch
from documents import SENTENCES

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

try:
    from ipex_llm.transformers.convert import _optimize_pre, _optimize_post
    IPEX_AVAILABLE = True
except ImportError:
    logging.warning("ipex-llm not detected. Running without optimizations.")
    IPEX_AVAILABLE = False


def generate_embeddings(model, texts, device):
    """
    Generates embeddings for the given texts using the specified model.

    Args:
        model: The SentenceTransformer model.
        texts: List of texts to generate embeddings for.
        device: Device to be used for inference (CPU/GPU).

    Returns:
        Normalized embeddings.
    """
    return model.encode(texts, normalize_embeddings=True, device=device)


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "BAAI/bge-large-en-v1.5"

    logging.info(f"Generating embeddings for {len(SENTENCES)} documents using device: {device}")

    # Load the model
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    # Apply optimizations only if running on CPU and ipex-llm is available
    if device.type == "cpu" and IPEX_AVAILABLE:
        model = _optimize_pre(model)
        model = _optimize_post(model)
        logging.info("Optimizations with ipex-llm applied.")

    # Generate embeddings and measure execution time
    start_time = time.time()
    embeddings = generate_embeddings(model, SENTENCES, device)
    end_time = time.time()

    logging.info(f"Time to generate embeddings: {end_time - start_time:.4f} seconds")