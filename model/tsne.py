from __future__ import print_function

import argparse

import scipy
import torch.backends.cudnn as cudnn

from VisualizeH.distance_distribution.intra_inter_distance import displt
from VisualizeH.tsne.tsne import plot_embedding_2d
from model import embed_net
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu or llcm]')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--optim', default='SGD', type=str, help='SGD,ADM')
parser.add_argument('--model_path', default='result/DJANet/save_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='result/DJANet/log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='result/DJANet/log/vis_log/', type=str, help='log save path')
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
#
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
    test_mode = [2, 1]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best_acc = 0  # best test accuracy
# start_epoch = 0
# pool_dim = 2048
# print('==> Building model..')
#
# if args.backbone == 'AGW':
#     embeding_dim = 2048
#     net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
# else:
#     embeding_dim = 2048
#     net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
#
# net.to(device)
# cudnn.benchmark = True
#
# checkpoint_path = args.model_path



if not os.path.isdir('result/DJANet/vis/save_tsne/{}'.format(args.dataset)):
    os.makedirs('result/DJANet/vis/save_tsne/{}'.format(args.dataset))


if not os.path.isdir('result/DJANet/vis/save_inttra_inter/{}'.format(args.dataset)):
    os.makedirs('result/DJANet/vis/save_inttra_inter/{}'.format(args.dataset))

result = scipy.io.loadmat('result/DJANet/vis/{}_result_{}_{}.mat'.format(args.dataset, test_mode[0], test_mode[1]))
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]

gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]


plot_embedding_2d(query_feature, query_label, args.dataset)
displt(query_feature, query_label, gallery_feature, gallery_label, args.dataset)



