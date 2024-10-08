import numpy as np
from PIL import Image
import pdb
import os
import random

data_path = '/root/autodl-tmp/dataset/SYSU-MM01/'
rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras = ['cam3','cam6']

# load id info
#'/root/HXC/reid/dataset/SYSU_MM01/exp/train_id.txt'
file_path_train = os.path.join(data_path,'exp/train_id.txt')
#'/root/HXC/reid/dataset/SYSU_MM01/exp/val_id.txt'
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    #[1,2,3,4,5,...]
    ids = [int(y) for y in ids[0].split(',')]
    #['0001','0002',...]
    id_train = ["%04d" % x for x in ids]
    
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    #[334,335,336,337,...]
    ids = [int(y) for y in ids[0].split(',')]
     #['0334','0335',...]
    id_val = ["%04d" % x for x in ids]
    
# combine train and val split   
id_train.extend(id_val) #id_val
#print(id_train.extend(id_val) )
files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        #'/root/HXC/reid/dataset/SYSU_MM01/cam6/0533'
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)
            
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)
#
#
# train_a = files_ir[:2000]
# train_b = files_rgb[:2000]

train_num = 10000
train_a = random.sample(files_ir, train_num)
train_b = random.sample(files_rgb, train_num)

test_num = 2000
# test_a = files_ir[2000:2500]
# test_b = files_rgb[2000:2500]

test_a = random.sample(files_ir, test_num)
test_b = random.sample(files_rgb, test_num)

save_path = '/root/autodl-tmp/4-re-id/img2img-turbo/data/my_Infrared2Visible'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 定义缩放尺寸
fix_image_width = 192
fix_image_height = 384
# 保存4位数字的文件名
for i, (a, b) in enumerate(zip(train_a, train_b)):
    img_a = Image.open(a)
    img_b = Image.open(b)
    # resize to 256x128
    img_a = img_a.resize((fix_image_width, fix_image_height))
    img_b = img_b.resize((fix_image_width, fix_image_height))
    # 图片名字4位数
    img_a.save(os.path.join(save_path, 'train_A', f'{str(i).zfill(4)}.jpg'))
    img_b.save(os.path.join(save_path, 'train_B', f'{str(i).zfill(4)}.jpg'))


for i, (a, b) in enumerate(zip(test_a, test_b)):
    img_a = Image.open(a)
    img_b = Image.open(b)
    img_a = img_a.resize((fix_image_width, fix_image_height))
    img_b = img_b.resize((fix_image_width, fix_image_height))
    # 图片名字4位数
    img_a.save(os.path.join(save_path, 'test_A', f'{str(i).zfill(4)}.jpg'))
    img_b.save(os.path.join(save_path, 'test_B', f'{str(i).zfill(4)}.jpg'))

# #files_rgb files_ir
# # relabel
# pid_container = set()
# #img_path：'/root/HXC/reid/dataset/SYSU_MM01/cam3/0533/0020.jpg'
# for img_path in files_ir:
#     #533
#     pid = int(img_path[-13:-9])
#     #{1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, ...,533}
#     pid_container.add(pid)
# #{1: 0, 2: 1, 4: 2, 5: 3, 7: 4, 8: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 18: 12, 19: 13, ...}
# pid2label = {pid:label for label, pid in enumerate(pid_container)}
# fix_image_width = 192
# fix_image_height = 384
