# Ollama Tutorial

## 🌟 Introduction

Ollama is an open-source tool that allows you to run large language models locally on your machine. This makes AI accessible without relying on cloud services or internet connectivity.

## ⬇️ Installation
```
# Linux/macOS one-line install
curl -fsSL https://ollama.com/install.sh | bash

# Verify installation
ollama --version
# Example output: ollama version 0.1.x
```

## 🛠️ Basic Usage
```
# download model


# Start a model (auto-downloads on first run)
ollama run deepseek-coder:7b

# Example interaction:
>>> Write a Python function to sort a list
# Model will return code implementation

# Exit interactive mode
/bye
```

## 🐍 Python Integration

```
# use pip to Install 
pip install ollama
```

```
# Example 

from ollama import chat

response = chat(
    model='deepseek-coder', 
    messages=[{'role': 'user', 'content': 'Explain recursion with an example'}]
)
print(response['message']['content'])

from ollama import chat

stream = chat(
    model='deepseek-coder',
    messages=[{'role': 'user', 'content': 'Write a Python quick sort implementation'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```
## 📁  Model Storage Configuration
Change default model storage location:
```
echo 'export OLLAMA_MODELS="/newdata/OneRAG/OneRAGModel"' >> ~/.bashrc
source ~/.bashrc
```

## 📊 Recommended Models
| Model | Best For |
|-------|----------|
| `mistral` | General purpose tasks |
| `llama2` | English conversations |
| `deepseek-coder` | Programming assistance |
| `qwen` | Chinese language tasks |
| `llava` | Multimodal (image + text) |
