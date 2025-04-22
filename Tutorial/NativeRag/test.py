from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image

# 加载模型和处理器
device = "cuda:0"
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16,  # 半精度节省显存
    device_map=device          # 自动分配设备（CPU/GPU）
)

# 图像和文本输入
image = Image.open("/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/OneRAG/Tutorial/NativeRag/dataset/imgs/R0000019.png")
question = "Question: What is the cat doing? Answer:"

# 处理并生成回答
inputs = processor(image, return_tensors="pt").to(device, torch.float16)
out = model.generate(**inputs, max_new_tokens=50)
answer = processor.decode(out[0], skip_special_tokens=True)
print("多模态问答结果[输入图像+文本]:")
print(answer) 

# from transformers import AutoProcessor, Blip2ForConditionalGeneration
# import torch
# from PIL import Image
# device = "cuda:0"
# # 加载模型和处理器
# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", 
#     torch_dtype=torch.float16,
#     device_map=device
# ).eval()  # 设置为评估模式

# # 图像预处理
# image = Image.open("/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/OneRAG/Tutorial/NativeRag/dataset/imgs/R0000019.png")
# inputs = processor(images=image, return_tensors="pt").to(model.device, torch.float16)

# # 提取图像特征
# with torch.no_grad():
#     # 通过视觉编码器获取特征
#     vision_outputs = model.vision_model(**inputs)
#     image_embeds = vision_outputs.last_hidden_state
#     print(image_embeds.shape)
#     # 池化处理（示例：取CLS令牌）
#     image_features = image_embeds[:, 0, :].cpu().numpy()

# print("图像特征维度:", image_features.shape)  # 输出示例: (1, 1408)