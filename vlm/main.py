import os
import time
import torch
import argparse
import logging
import requests
from PIL import Image

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

from transformers import AutoProcessor

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict tokens using `generate()` API for any LLM."
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="microsoft/Phi-3.5-vision-instruct",
        help="The Hugging Face repo ID for the model to be downloaded, "
             "or the path to the Hugging Face checkpoint folder.",
    )
    parser.add_argument(
        "--image-url-or-path",
        type=str,
        default="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg",
        help="The URL or path to the image to infer.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Generate a detailed caption for the image.",
        help="Prompt to infer.",
    )
    parser.add_argument(
        "--n-predict", type=int, default=1024, help="Maximum tokens to predict."
    )

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image_path = args.image_url_or_path

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set model loading parameters
    opt_parameters = {}
    if IPEX_OPT:
        opt_parameters["load_in_low_bit"] = "sym_int4"
        opt_parameters["modules_to_not_convert"] = ["vision_embed_tokens"]
        opt_parameters["optimize_model"] = True
        logging.info("Optimized parameters for ipex-llm are being used.")
    else:
        opt_parameters["device_map"] = device
        logging.info("Using default device mapping for model loading.")

    # Load model and processor
    logging.info("Loading model and processor...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        _attn_implementation="eager",
        **opt_parameters,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    logging.info("Model and processor loaded successfully.")

    # Construct the prompt
    messages = [
        {"role": "user", "content": "<|image_1|>\n{prompt}".format(prompt=args.prompt)},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Load the image
    try:
        if os.path.exists(image_path):
            logging.info(f"Loading image from local path: {image_path}")
            image = Image.open(image_path)
        else:
            logging.info(f"Loading image from URL: {image_path}")
            image = Image.open(requests.get(image_path, stream=True).raw)
    except Exception as e:
        logging.error(f"Failed to load image. Error: {e}")
        raise

    # Perform inference
    logging.info("Starting inference...")
    with torch.inference_mode():
        inputs = processor(prompt, [image], return_tensors="pt")
        inputs = inputs.to(device) if device.type != "cpu" else inputs

        start_time = time.time()
        output = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=1,
            do_sample=False,
            max_new_tokens=args.n_predict,
            temperature=0.0,
        )
        end_time = time.time()
        inference_time = end_time - start_time
        logging.info(f"Inference completed in {inference_time:.2f} seconds.")

        output = output[:, inputs['input_ids'].shape[1]:]

        # Decode output
        output_str = processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        num_generated_tokens = len(processor.tokenizer(output_str[0])["input_ids"])
        tokens_per_second = num_generated_tokens / inference_time

        # Log token statistics
        logging.info(f"Generated {num_generated_tokens} tokens at {tokens_per_second:.2f} tokens per second.")

        # Log results
        logging.info("-" * 20 + " Prompt " + "-" * 20)
        logging.info(f"Message: {messages}")
        logging.info(f"Image link/path: {image_path}")
        logging.info("-" * 20 + " Output " + "-" * 20)
        logging.info(output_str)
