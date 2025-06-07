```markdown
# Ollama 使用教程

## 🌟 简介

Ollama 是一款开源工具，可在本地计算机上运行大型语言模型。

## ⬇️ 安装指南

```bash
# Linux/macOS 一键安装
curl -fsSL https://ollama.com/install.sh | bash

# 验证安装
ollama --version
# 预期输出示例：ollama version 0.1.x
```

## 🛠️ 基础使用

```bash
# 启动模型（首次运行会自动下载）
ollama run deepseek-coder:7b

# 交互示例：
>>> 用Python写一个列表排序函数
# 模型将返回代码实现

# 退出交互模式
/bye
```

## 🐍 Python 集成

```bash
# 使用pip安装
pip install ollama
```

```python
# 基础调用示例
from ollama import chat

response = chat(
    model='deepseek-coder', 
    messages=[{'role': 'user', 'content': '用示例解释递归的概念'}]
)
print(response['message']['content'])

# 流式调用示例
from ollama import chat

stream = chat(
    model='deepseek-coder',
    messages=[{'role': 'user', 'content': '用Python实现快速排序算法'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

## 📁 模型存储配置
修改默认模型存储路径：

```bash
echo 'export OLLAMA_MODELS="/newdata/OneRAG/OneRAGModel"' >> ~/.bashrc
source ~/.bashrc
```

## 📊 推荐模型列表

| 模型名称 | 最佳适用场景 |
|---------|------------|
| `mistral` | 通用任务处理 |
| `llama2` | 英文对话交流 |
| `deepseek-coder` | 编程辅助开发 |
| `qwen` | 中文语言处理 |
| `llava` | 多模态任务（图像+文本） |
```

