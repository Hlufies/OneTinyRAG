import os
os.environ["TRANSFORMERS_CACHE"] = "/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OLLAMA_MODELS"] = "/newdata/HTY/BIO_LLM/DS_70b/models"
os.environ["OLLAF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/models"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
print(sys.executable)  # 应显示虚拟环境路径而非系统路径

import os
os.environ["TRANSFORMERS_CACHE"] = "/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OLLAMA_MODELS"] = "/newdata/HTY/BIO_LLM/DS_70b/models"
os.environ["OLLAF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/models"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoProcessor, Blip2ForConditionalGeneration,  Blip2ForImageTextRetrieval
import torch
from PIL import Image

# 加载模型和处理器
device = "cuda:0"
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir="/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/models")
model = Blip2ForImageTextRetrieval.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    # torch_dtype=torch.float16,  # 半精度节省显存
    device_map=device,          # 自动分配设备（CPU/GPU）
    cache_dir="/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/models"  # 指定缓存目录
)
image = Image.open("/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/OneRAG/Tutorial/NativeRAG/dataset/imgs/R0000019.png")
question = "Question: What is the cat doing? Answer:"

inputs = processor(image, return_tensors="pt").to(device)
img = model.vision_model(**inputs)
image_embeds = img[0]
image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

print(image_embeds)
print(image_embeds.shape)

print(image_attention_mask)
print(image_attention_mask.shape)
query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
query_outputs = model.qformer(
    query_embeds=query_tokens,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_attention_mask,
    return_dict=None,
)
tokenizer = processor.tokenizer
inputs = tokenizer(
    question,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=32
)

input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
input_ids = input_ids.to(model.device)
attention_mask = attention_mask.to(model.device)
query_embeds = model.embeddings(input_ids=input_ids)