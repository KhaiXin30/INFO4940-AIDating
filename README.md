# INFO4940-AIDating

## Setup

### Download

This project requires the Llama 3.1 8B. Download it using:

```bash
wget "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" -L -o llama-3.1-8b.gguf
```
or run

```bash
curl "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" -L -o llama-3.1-8b.gguf (if on Mac)
```

Then install other dependencies:
```bash
pip install -r requirements.txt
```


### Option 1: Command Line Interface:

First run, CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python if you are on a Mac (Apple Silicon). If on Windows, run CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python. If neither of these works, try running pip install llama-cpp-python.

##### In the terminal: 
```bash
python model3.py
```

### Option 2: Web UI (Recommended)

Run the Streamlit web interface:

```bash
streamlit run app.py
```
This will start a local web server and open the application in your browser at `http://localhost:8501`.