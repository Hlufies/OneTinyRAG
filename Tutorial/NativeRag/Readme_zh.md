
# Tutorial-NativeRAG

基于 LangChain+Deepseek+Faiss 的RAG的Tutorial。  

## Todo
- [2025.4.10] 🔥多格式文件自动处理功能
- [2025.4.10] 🔥Ollama + Deepseek本地化部署
- [2025.4.10] 🔥扩展SemanticChunker（Spacy和NLTK）
  

## 快速开始
### 安装依赖
```bash
conda activate DeepseekRag
pip install -r requirements.txt
```

### 运行示例
```bash
python app.py
```

## 项目结构

```
/NativeRag/
├── app.py                 # 主程序入口
├── Readme.md              # 英文文档
├── Readme_zh.md           # 中文文档
├── Indexer/               # 索引处理模块
│   ├── __init__.py
│   ├── Indexer.py         # 索引管理核心
│   ├── Embedder.py        # 嵌入模型抽象
│   ├── DataProcessor.py   # 文档处理器
│   └── Chunker.py         # 文本分块策略
├── Generator/             # 结果生成模块
│   └── Generator.py       
├── Retriever/             # 检索模块
│   └── Retriever.py
└── config/                # 配置文件目录
    └── config1.json
```

## 核心组件

```python
indexer = Indexer(config)
retriever = Retriever(indexer.embedder.embedder, index, config)
generator = Generator(indexer.embedder, config)
```

## 配置说明
通过`config/config1.json`可配置：


## 扩展指南
1. 在`Indexer.py`中注册
2. 在`DataProcessor.py`中扩展新文档处理器
3. 在`Chunker.py`中扩展新文本切分器
4. 在`Embedder.py`中扩展新embedder
5. 在`Retriever.py`中注册
6. 在`Retriever.py`中扩展新检索器
7. 在`Generator.py`中注册
8. 在`Generator.py`中扩展新的生成器

