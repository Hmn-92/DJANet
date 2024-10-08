import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from src.cyclegan_turbo import CycleGAN_Turbo
from src.my_utils.training_utils import build_transform


if __name__ == "__main__":

    # root = "/root/autodl-tmp/dataset/RegDB/Thermal"
    #
    # # 遍历文件夹下的所有图像文件
    # for dirpath, dirnames, files in os.walk(root):
    #     for dir in dirnames:
    #         dir_root = os.path.join(root, dir)
    #         for file in os.listdir(dir_root):
    #             input_image_path = os.path.join(root, dir, file)
    #             input_image = Image.open(input_image_path).convert('RGB')

    pretrained_name = None
    pretrained_path = "output/cyclegan_turbo/my_Infrared2Visible/checkpoints/model_20501.pkl"
    image_prep = "resize_192x384"

    input_image_path = "/root/autodl-tmp/dataset/LLCM/nir/0000/0000_c05_s171429_f12990_nir.jpg"
    prompt = "picture of a visible"
    direction = "a2b"
    output_dir = "/root/autodl-tmp/dataset/LLCM/nir_generated"

    # initialize the model
    model = CycleGAN_Turbo(pretrained_name=pretrained_name, pretrained_path=pretrained_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    T_val = build_transform(image_prep)

    root = "/root/autodl-tmp/dataset/RegDB/Thermal"
    output_dir = "/root/autodl-tmp/dataset/RegDB/Thermal_generated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 遍历文件夹下的所有图像文件
    # 遍历文件夹下的所有图像文件
    for dirpath, dirnames, files in os.walk(root):
        for dir in dirnames:
            dir_root = os.path.join(root, dir)
            for file in os.listdir(dir_root):
                input_image_path = os.path.join(root, dir, file)
                input_image = Image.open(input_image_path).convert('RGB')
                # translate the image
                with torch.no_grad():
                    input_img = T_val(input_image)
                    x_t = transforms.ToTensor()(input_img)
                    x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
                    output = model(x_t, direction=direction, caption=prompt)

                output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
                output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

                # save the output image
                bname = os.path.basename(input_image_path)
                save_output_dir = os.path.join(output_dir, dir)
                os.makedirs(save_output_dir, exist_ok=True)
                output_pil.save(os.path.join(save_output_dir, bname))


    #
    # input_image = Image.open(input_image_path).convert('RGB')
    # # translate the image
    # with torch.no_grad():
    #     input_img = T_val(input_image)
    #     x_t = transforms.ToTensor()(input_img)
    #     x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
    #     output = model(x_t, direction=direction, caption=prompt)
    #
    # output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    # output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)
    #
    # # save the output image
    # bname = os.path.basename(input_image_path)
    # os.makedirs(output_dir, exist_ok=True)
    # output_pil.save(os.path.join(output_dir, bname))
