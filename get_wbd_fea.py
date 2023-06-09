#%% 用 ~/miniconda3/envs/py37_torch1_7/bin/python 执行
# 直接用wbdataset读取，感觉有点慢
import torch
import clip
from PIL import Image
from torchvision import transforms
from itertools import islice
import time
import webdataset as wds


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

sharedurl = "/data2/opensource/LAION-400M/laion400m-data/{00000..01002}.tar"

dataset = (
    wds.WebDataset(sharedurl)
    .decode("pil")
    .to_tuple("jpg;png", "json")
)

with torch.no_grad():
    index_all = 0
    # 外部tar文件按顺序，内部不是，但顺序每次一样，文件名序号是整体增加不会随着换tar从头开始
    batch_size = 1024
    img_tensor_batch = torch.zeros([batch_size,3,224,224])
    index = 0
    t1 = time.time()
    for image, data in islice(dataset, 0, 20000):
        # print(index_all, data['key'])
        image = preprocess(image)
        img_tensor_batch[index] = image
        index_all += 1
        index += 1
        # 达到bs开始推理
        if index == batch_size:
            image_features = model.encode_image(img_tensor_batch.to(device))
            index = 0
            print(image_features.shape)
            print(f"average time per img is:{(time.time()-t1)/index_all} second")

