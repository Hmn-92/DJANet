from __future__ import print_function

import argparse
import time

import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt
from data_manager import *
from utils import *
import scipy
import cv2

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu or llcm]')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--optim', default='SGD', type=str, help='SGD,ADM')
parser.add_argument('--model_path', default='result/DJANNet/save_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='result/DJANNet/log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='result/DJANNet/log/vis_log/', type=str, help='log save path')
parser.add_argument('--loss_tri', default='', type=str, help='')
parser.add_argument('--lr_scheduler', default='step', type=str, help='step or consine')
parser.add_argument('--backbone', default='AGW', type=str, help='AGW')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035/0.0001 for adam,sgd 0.1')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.7, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=2, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=int, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args()

dataset = args.dataset

if dataset == 'sysu':
    data_path = '/dataset/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
elif dataset == 'regdb':
    data_path = '/dataset/RegDB/'
    n_class = 206
    test_mode = [2, 1]
elif dataset == 'llcm':
    data_path = '/dataset/LLCM/'
    n_class = 713
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;

def process_gallery_sysu(data_path, mode=2, trial=0, relabel=False):
    if mode == 2:
        ir_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 1:
        ir_cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_query_sysu(data_path, mode=1, relabel=False):
    if mode == 1:
        ir_cameras = ['cam3', 'cam6']
    elif mode == 2:
        ir_cameras = ['cam1', 'cam2', 'cam4', 'cam5']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_llcm(img_dir, mode=1):
    if mode == 1:
        input_data_path = os.path.join(data_path, 'idx/test_vis.txt')
    elif mode == 2:
        input_data_path = os.path.join(data_path, 'idx/test_nir.txt')

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        file_cam = [int(s.split('c0')[1][0]) for s in data_file_list]

    return file_image, np.array(file_label), np.array(file_cam)

end = time.time()
if dataset == 'sysu':
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=test_mode[0])
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=test_mode[1], trial=0)

elif dataset == 'regdb':
    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

elif dataset == 'llcm':
    # testing set
    query_img, query_label, query_cam = process_query_llcm(
        data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(
        data_path, mode=test_mode[0], trial=0)


if not os.path.isdir('result/DJANet/vis/save_ranking/{}_{}_{}'.format(args.dataset, test_mode[0], test_mode[1])):
    os.makedirs('result/DJANet/vis/save_ranking/{}_{}_{}'.format(args.dataset, test_mode[0], test_mode[1]))


#######################################################################
# Evaluate
for j in range(8680):
    if j % 10 == 0:
        print(j)
        result = scipy.io.loadmat('result/DJANet/vis/{}_result_{}_{}.mat'.format(args.dataset, test_mode[0], test_mode[1]))
        query_feature = torch.FloatTensor(result['query_f'])
        query_cam = result['query_cam'][0]
        query_label = result['query_label'][0]

        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_cam = result['gallery_cam'][0]
        gallery_label = result['gallery_label'][0]

        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        #####################################################################
        # Show result
        def imshow(path, title=None):
            """Imshow for Tensor."""
            im = plt.imread(path)
            im = cv2.resize(im, (args.img_w, args.img_h))
            plt.imshow(im)
            if title is not None:
                plt.title(title)
            plt.pause(0.001)  # pause a bit so that plots are updated

        #######################################################################
        # sort the images
        def sort_img(qf, ql, qc, gf, gl, gc):
            query = qf.view(-1, 1)
            score = torch.mm(gf, query)
            score = score.squeeze(1).cpu()
            score = score.numpy()
            # predict index
            index = np.argsort(score)  # from small to large
            index = index[::-1]
            # good index
            query_index = np.argwhere(gl == ql)
            # same camera
            camera_index = np.argwhere(gc == qc)

            junk_index1 = np.argwhere(gl == -1)
            junk_index2 = np.intersect1d(query_index, camera_index)
            junk_index = np.append(junk_index2, junk_index1)

            mask = np.in1d(index, junk_index, invert=True)
            index = index[mask]
            return index


        i = j
        index = sort_img(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)

        query_path = query_img[i]
        query_label = query_label[i]
        print(query_path)
        print('Top 10 images are as follow:')
        # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        imshow(query_path, 'query')
        for i in range(10):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            img_path = gall_img[index[i]]
            label = gallery_label[index[i]]

            if int(label) == int(query_label):
                ax.set_title('%d' % (i + 1), color='green')
                imshow(img_path, 'True')
                print(img_path, 'True')
            else:
                ax.set_title('%d' % (i + 1), color='red')
                imshow(img_path, 'False')
                print(img_path, 'False')

        fig.savefig("result/DJANet/vis/save_ranking/{}_{}_{}/show".format(args.dataset, test_mode[0], test_mode[1]) + str(j) + ".jpg")
