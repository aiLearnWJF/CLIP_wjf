#%% 用 ~/miniconda3/envs/py37_torch1_7/bin/python 执行
# 直接用wbdataset读取，感觉有点慢
import torch
import clip
from PIL import Image
from torchvision import transforms
from itertools import islice
import time
import webdataset as wds
import os
import sys

batch_size = 3072

device = "cuda:%s"%(sys.argv[1]) if torch.cuda.is_available() else "cpu"
print("using gpu:",device)

sharedurl = sys.argv[2]
print("process wds path:",sharedurl)
# "/data2/opensource/LAION-400M/laion400m-data/{00000..00002}.tar"


model, preprocess = clip.load("ViT-B/32", device=device)

preproc = transforms.Compose([
    preprocess,
])

dataset = (
    wds.WebDataset(sharedurl)
    .decode("pil")
    .to_tuple("jpg;png", "json")
    .map_tuple(preproc)
)


dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size), num_workers=4, batch_size=None)

with torch.no_grad():
    index_all = 0
    # 外部tar文件按顺序，内部不是，但顺序每次一样，文件名序号是整体增加不会随着换tar从头开始
    # batch_size = 1024
    # img_tensor_batch = torch.zeros([batch_size,3,224,224])
    index = 0
    t1 = time.time()
    for images,targets in dataloader:
        print("processing indexall:",index_all,targets[0]['key'],images.shape[0])
        img_tensor_batch = torch.Tensor(images)
        # image_features = model.encode_image(img_tensor_batch)
        image_features = model.encode_image(img_tensor_batch.to(device))
        index_all += batch_size
        index += 1

        # save bs
        f = open("%s/%d_key.txt"%(sys.argv[3],index),'w')
        torch.save(image_features, "%s/%d_fea.pt"%(sys.argv[3],index))
        for i in targets:
            f.write(i['key'] + "\n")
        f.flush()
        f.close()

        print(f"average time per img is:{(time.time()-t1)/(index_all)} second")


    print(f"total time is:{time.time()-t1} second")
