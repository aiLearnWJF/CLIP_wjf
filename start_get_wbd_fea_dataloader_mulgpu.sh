export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="/vehicle/yckj3860/code/CLIP_wjf/clip"

pkill -9 python
rm -rf pt1 pt2 pt3 pt4
mkdir pt1
mkdir pt2
mkdir pt3
mkdir pt4
nohup /vehicle/yckj3860/miniconda3/envs/py37_torch1_7/bin/python get_wbd_fea_dataloader.py 1 "/data2/opensource/LAION-400M/laion400m-data/{00000..20000}.tar" "pt1" >pt1.log 2>&1  &
nohup /vehicle/yckj3860/miniconda3/envs/py37_torch1_7/bin/python get_wbd_fea_dataloader.py 2 "/data2/opensource/LAION-400M/laion400m-data/{20001..41407}.tar" "pt2" >pt2.log 2>&1  &
nohup /vehicle/yckj3860/miniconda3/envs/py37_torch1_7/bin/python get_wbd_fea_dataloader.py 3 "/data2/opensource/LAION-400M/laion400m-data-round2/{00000..25797}.tar" "pt3" >pt3.log 2>&1  &
nohup /vehicle/yckj3860/miniconda3/envs/py37_torch1_7/bin/python get_wbd_fea_dataloader.py 4 "/data2/opensource/LAION-400M/laion400m-data-round3/{00000..13985}.tar" "pt4" >pt4.log 2>&1  &