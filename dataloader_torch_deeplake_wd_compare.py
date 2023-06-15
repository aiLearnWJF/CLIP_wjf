#%% 用 ~/miniconda3/envs/py37_torch1_7/bin/python 执行
import deeplake
from PIL import Image
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms, models
import time

# ┌────────────────────────────────────────────────────────────────────────┐
# │  创建本地deeplake数据集          
# └────────────────────────────────────────────────────────────────────────┘
# ds = deeplake.empty('./animals_deeplake',overwrite=True) # Create the dataset locally
# # Find the class_names and list of files that need to be uploaded
# dataset_folder = './animals'

# # Find the subfolders, but filter additional files like DS_Store that are added on Mac machines.
# class_names = [item for item in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, item))]

# files_list = []
# for dirpath, dirnames, filenames in os.walk(dataset_folder):
#     for filename in filenames:
#         files_list.append(os.path.join(dirpath, filename))


# with ds:
#     # Create the tensors with names of your choice.
#     ds.create_tensor('images', htype = 'image', sample_compression = 'jpeg')
#     ds.create_tensor('labels', htype = 'class_label', class_names = class_names)
    
#     # Add arbitrary metadata - Optional
#     ds.info.update(description = 'My first Deep Lake dataset')
#     ds.images.info.update(camera_type = 'SLR')

# with ds:
#     # Iterate through the files and append to Deep Lake dataset
#     for file in files_list:
#         label_text = os.path.basename(os.path.dirname(file))
#         label_num = class_names.index(label_text)
        
#         #Append data to the tensors
#         ds.append({'images': deeplake.read(file), 'labels': np.uint32(label_num)})

# # Image.fromarray(ds.images[0].numpy())

# ds.summary()
# ds.visualize()

# ┌────────────────────────────────────────────────────────────────────────┐
# │  datawebset 转 deeplake数据集          
# └────────────────────────────────────────────────────────────────────────┘
import deeplake
import webdataset as wds
from deeplake.core.sample import Sample

raw_data = 's3://non-hub-datasets-n/laion400m-data/'

ds = deeplake.empty('s3://hub-2.0-datasets-n/laion400m-data-test/')
ds.create_tensor('image', htype='image', sample_compression='jpeg')
ds.create_tensor('caption', htype='text')

files_list = []
for i in range(41408):
    files_list.append(raw_data + str(i).zfill(5) + '.tar')

@deeplake.compute
def process(file, ds_out):
    url = 'pipe:aws s3 cp ' + file + " -"
    with ds:
        wd = wds.WebDataset(url).to_tuple('jpg', 'txt', 'json')
        for img, txt, _ in wd:
            img = Sample(buffer=img, compression='jpeg')
            t = txt.decode()
            if len(img.shape) == 2:
                continue
            ds_out.image.append(img)
            ds_out.caption.append(t)

process().eval(files_list, ds, scheduler='processed', num_workers=6)

#%% 
# ┌────────────────────────────────────────────────────────────────────────┐
# │  开始ds dataloader测试          
# └────────────────────────────────────────────────────────────────────────┘
batch_size = 64
ds = deeplake.load('animals_deeplake')
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224]),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

deeplake_loader = ds.pytorch(num_workers=4, batch_size=64, transform={
                        'images': tform, 'labels': None}, shuffle=True)
count_idx = 0
t1 = time.time()
for i, data in enumerate(deeplake_loader):
    images, labels = data['images'], data['labels']
    count_idx += labels.shape[0]
    if count_idx%2560 == 0:
        print(f"process img nums:{count_idx},average nums per second is: {count_idx/(time.time() - t1)}")

#%% 
# ┌────────────────────────────────────────────────────────────────────────┐
# │  开始torch dataloader测试          
# └────────────────────────────────────────────────────────────────────────┘
# batch_size = 64
# num_workers = 4
# tform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize([224,224]),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
# ])
# # ImageFolder 通用的加载器
# dataset = torchvision.datasets.ImageFolder("animals/", transform=tform)
# # 构建可迭代的数据装载器
# inputs = torch.utils.data.DataLoader(dataset=dataset, batch_size = batch_size, shuffle=True, num_workers = num_workers)
# count_idx = 0
# t1 = time.time()
# for data, label in inputs:
#     count_idx += label.shape[0]
#     if count_idx%2560 == 0:
#         print(f"process img nums:{count_idx},average nums per second is: {count_idx/(time.time() - t1)}")


# %%
# ┌────────────────────────────────────────────────────────────────────────┐
# │  图片复制删除工具          
# └────────────────────────────────────────────────────────────────────────┘
# import glob
# import shutil
# import os
# for i in range(10000):
#     shutil.copy("animals/cats/image_1.jpg", "animals/cats/image_%d.jpg"%(i+5))
#     shutil.copy("animals/dogs/image_3.jpg", "animals/dogs/image_%d.jpg"%(i+5))
    # os.remove("animals/cats/image_%d.jpg"%(i+5))
    # os.remove("animals/dogs/image_%d.jpg"%(i+5))