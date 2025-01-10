# Intel IPEX-LLM Benchmarking on 5th Gen Xeon Processors

## Overview
The main goal of this repository is to evaluate the performance of Intel's 5th Generation Xeon "Emerald Rapids" processors in multimodal Retrieval-Augmented Generation (RAG) scenarios using CPUs. Specifically, this benchmarking focuses on three key models that form the multimodal pipeline:

- **Embeddings (BAAI/bge-large-en-v1.5):** For generating high-quality semantic text representations.
- **Large Language Model (Llama-3.2-1B-Instruct):** A compact instruction-following LLM.
- **Vision Language Model (Phi-3.5-vision-instruct):** Handles tasks that combine visual and textual data.

This repository provides scripts and instructions to measure inference times for these models on both CPU and GPU environments.

---

## Getting Started

### Hugging Face Configuration
To use the **Llama-3.2-1B-Instruct** model, you first need to obtain access via Hugging Face. Follow these steps:

1. Request access to the model from [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
2. Once access is granted, generate a user access token to authorize model downloads. Refer to the [Hugging Face documentation](https://huggingface.co/docs/hub/security-tokens) for detailed instructions.

Create a .env file in the llm folder with the following content HF_TOKEN=__REPLACE_TOKEN__

### Main Dependencies

Create a Python 3.11 environment using your preferred environment manager and ensure `pip` is updated to version 24.2 or later:

```bash
python -m pip install --upgrade pip
```


## Benchmarking Models

### 1. **Embeddings**

#### CPU Environment

Install the required dependencies for CPU-based inference:

```bash
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_embeddings_cpu.txt
```

#### GPU Environment

Install the required dependencies for GPU-based inference:

```bash
pip install -r requirements_embeddings_gpu.txt
```

#### Script Execution

To measure inference time for embeddings, run the following script:

```bash
python embeddings/main.py
```

### 2. **Large Language Model and Vision Language Model**

#### CPU Environment

Install the required dependencies for CPU-based inference:

```bash
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_cpu.txt
```

#### GPU Environment

Install the required dependencies for GPU-based inference:

```bash
pip install -r requirements_gpu.txt
```

#### Script Execution

To measure inference time for the large language model, execute the script:

```bash
python llm/main.py
```

To measure inference time for the vision language model, execute the script:

```bash
python vlm/main.py
```

**Note**: In linux environments execute the following commands before script execution:

```bash
> source ipex-llm-init 
> numactl -C 0-NUM_PROCESSORS -m 0 python SCRIPT_PATH
```

---

## Results and Outputs
The benchmarking scripts will output inference time metrics for each model. These metrics can be used to compare CPU and GPU performance under different configurations.



## Additional Resources

- [Intel IPEX LLM Documentation](https://github.com/intel-analytics/ipex-llm)
- [Hugging Face Documentation](https://huggingface.co/docs)

For further assistance, please create an issue in this repository.


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

