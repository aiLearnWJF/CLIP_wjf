#%% 用 ~/miniconda3/envs/py37_torch1_7/bin/python 执行
import torch
import clip
from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:1"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["Sachin Tendulkar Thought Sushant Singh Rajput Was a Cricketer", "raisins blancs sur un plat by guillaume romain fouace", "Pigment Grinding를 위한 높은 Efficiency Laboratory Ball Mill"]).to(device)

with torch.no_grad():
    image_features = torch.load("pt1/1_fea.pt")[:3].to(device)
    text_features = model.encode_text(text)
    matrix = torch.matmul(image_features, text_features.T)
    print(matrix)
    

# print(clip.available_models())