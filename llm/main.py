import torch
import time
import argparse
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

IPEX_OPT = False

try:
    from ipex_llm.transformers import AutoModelForCausalLM
    IPEX_OPT = True
    logging.info("ipex-llm optimizations are enabled.")
except ImportError:
    from transformers import AutoModelForCausalLM
    logging.warning("ipex-llm not detected. Running without optimizations.")

from transformers import AutoTokenizer

DEFAULT_SYSTEM_PROMPT = """\
"""


def get_prompt(user_input: str, chat_history: list[tuple[str, str]], system_prompt: str) -> str:
    """
    Constructs a prompt for the model based on user input, chat history, and system prompt.

    Args:
        user_input: The input provided by the user.
        chat_history: A list of tuples representing the chat history.
        system_prompt: The system-level prompt.

    Returns:
        A formatted string prompt for the model.
    """
    prompt_texts = [f"<|begin_of_text|>"]

    if system_prompt:
        prompt_texts.append(
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        )

    for history_input, history_response in chat_history:
        prompt_texts.append(
            f"<|start_header_id|>user<|end_header_id|>\n\n{history_input.strip()}<|eot_id|>"
        )
        prompt_texts.append(
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{history_response.strip()}<|eot_id|>"
        )

    prompt_texts.append(
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return "".join(prompt_texts)


if __name__ == "__main__":
    # Parse command-line arguments and load env variables.
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for Llama3.2 model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The Hugging Face repo ID for the model (e.g., `meta-llama/Meta-Llama-3.2-1B-Instruct`) "
             "or the path to the Hugging Face checkpoint folder.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is Artificial Intelligence? Please, provide a detailed description.",
        help="Prompt to infer.",
    )
    parser.add_argument(
        "--n-predict", type=int, default=1024, help="Maximum tokens to predict."
    )

    args = parser.parse_args()
    load_dotenv()
    model_path = args.repo_id_or_model_path

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set model loading parameters
    opt_parameters = {}

    if IPEX_OPT:
        opt_parameters["load_in_low_bit"] = "sym_int4"
        opt_parameters["optimize_model"] = True
        logging.info("Optimized parameters for ipex-llm are being used.")
    else:
        opt_parameters["device_map"] = device
        logging.info("Using default device mapping for model loading.")

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **opt_parameters,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logging.info("Model and tokenizer loaded successfully.")

    # Perform inference
    with torch.inference_mode():
        prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(device) if device.type != "cpu" else input_ids

        logging.info("Starting inference...")
        start_time = time.time()
        output = model.generate(input_ids, max_new_tokens=args.n_predict)
        end_time = time.time()
        inference_time = end_time - start_time
        logging.info(f"Inference completed in {inference_time:.2f} seconds.")

        # Decode output
        output_str = tokenizer.decode(output[0], skip_special_tokens=False)

        # Count the number of tokens generated
        num_generated_tokens = len(tokenizer(output_str)["input_ids"])
        tokens_per_second = num_generated_tokens / inference_time

        # Log token statistics
        logging.info(f"Generated {num_generated_tokens} tokens at {tokens_per_second:.2f} tokens per second.")

        # Log results
        logging.info("-" * 20 + " Prompt " + "-" * 20)
        logging.info(prompt)
        logging.info("-" * 20 + " Output (skip_special_tokens=False) " + "-" * 20)
        logging.info(output_str)
