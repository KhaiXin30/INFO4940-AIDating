# INFO4940-AIDating

## Setup

### Download the Model

This project requires the Qwen2-7B-Instruct model. Download it using:

wget "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_0.gguf" -O qwen2-7b-instruct-q4_0.gguf

or run

curl "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_0.gguf" -O qwen2-7b-instruct-q4_0.gguf (if on Mac)

### To run

First run, CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python if you are on a Mac (Apple Silicon). If on Windows, run CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python. If neither of these works, try running pip install llama-cpp-python.

In the terminal: python model2.py
