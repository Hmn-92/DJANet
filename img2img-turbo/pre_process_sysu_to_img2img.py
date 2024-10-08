import numpy as np
from PIL import Image
import pdb
import os
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from src.cyclegan_turbo import CycleGAN_Turbo
from src.my_utils.training_utils import build_transform

pretrained_name = None
pretrained_path = "output/cyclegan_turbo/my_Infrared2Visible/checkpoints/model_20501.pkl"
image_prep = "no_resize"


# initialize the model
model = CycleGAN_Turbo(pretrained_name=pretrained_name, pretrained_path=pretrained_path)
model.eval()
model.unet.enable_xformers_memory_efficient_attention()

T_val = build_transform(image_prep)

data_path = '/root/autodl-tmp/dataset/SYSU-MM01/'
save_data_path = '/root/autodl-tmp/dataset/SYSU-MM01_generate_1/'
os.makedirs(save_data_path, exist_ok=True)
rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

# load id info
# '/root/HXC/reid/dataset/SYSU_MM01/exp/train_id.txt'
file_path_train = os.path.join(data_path, 'exp/train_id.txt')
# '/root/HXC/reid/dataset/SYSU_MM01/exp/val_id.txt'
file_path_val = os.path.join(data_path, 'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    # [1,2,3,4,5,...]
    ids = [int(y) for y in ids[0].split(',')]
    # ['0001','0002',...]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    # [334,335,336,337,...]
    ids = [int(y) for y in ids[0].split(',')]
    # ['0334','0335',...]
    id_val = ["%04d" % x for x in ids]

# combine train and val split
id_train.extend(id_val)  # id_val
# print(id_train.extend(id_val) )
files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        # '/root/HXC/reid/dataset/SYSU_MM01/cam6/0533'
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in ir_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)
#
# files_rgb files_ir
# relabel
pid_container = set()
# img_path：'/root/HXC/reid/dataset/SYSU_MM01/cam3/0533/0020.jpg'
for img_path in files_ir:
    # 533
    pid = int(img_path[-13:-9])
    # {1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, ...,533}
    pid_container.add(pid)
# {1: 0, 2: 1, 4: 2, 5: 3, 7: 4, 8: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 18: 12, 19: 13, ...}
pid2label = {pid: label for label, pid in enumerate(pid_container)}
fix_image_width = 192
fix_image_height = 384


# train_image
def read_imgs(train_image, files_rgb):
    train_img = []
    train_label = []
    # img_path：'/root/HXC/reid/dataset/SYSU_MM01/cam3/0533/0020.jpg'
    if files_rgb in "files_rgb":
        prompt = "picture of a infrared"
        direction = "b2a"
    else:
        prompt = "picture of a visible"
        direction = "a2b"
    for img_path in train_image:
        # img
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        input_image = img.resize((fix_image_width, fix_image_height), Image.LANCZOS)
        # pix_array.shape
        # (288, 144, 3)
        # translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            output = model(x_t, direction=direction, caption=prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)

        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)
        # 另存为save_path 下的对应目录文件，如果不存在则创建
        save_path = os.path.join(save_data_path, img_path.split('/')[-3], img_path.split('/')[-2])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_img_path = os.path.join(save_path, img_path.split('/')[-1])

        save_pil = output_pil.resize((img.width, img.height), Image.LANCZOS)
        save_pil.save(save_img_path)

        pix_array = np.array(output_pil)
        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)
    return np.array(train_img), np.array(train_label)


# rgb imges
train_img, train_label = read_imgs(files_rgb, "files_rgb")
np.save(save_data_path + 'train_rgb2ir_resized_img.npy', train_img)
np.save(save_data_path + 'train_rgb2ir_resized_label.npy', train_label)

# ir imges
# files_ir:

train_img, train_label = read_imgs(files_ir, "files_ir")
# train_img:
# train_label:array([  0,   0,   0, ..., 394, 394, 394])
np.save(save_data_path + 'train_ir2rgb_resized_img.npy', train_img)
np.save(save_data_path + 'train_ir2rgb_resized_label.npy', train_label)
print(save_data_path + 'train_ir_resized_label.npy')
