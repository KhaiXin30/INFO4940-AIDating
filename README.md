# INFO4940-AIDating

## Setup

### Download the Model

This project requires the Qwen2-7B-Instruct model. Download it using:

wget "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" -L -o llama-3.1-8b.gguf

or run

curl "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" -L -o llama-3.1-8b.gguf (if on Mac)

### To run

First run, CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python if you are on a Mac (Apple Silicon). If on Windows, run CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python. If neither of these works, try running pip install llama-cpp-python.

In the terminal: python model2.py
