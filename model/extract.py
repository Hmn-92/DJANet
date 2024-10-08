from __future__ import print_function

import argparse
import time

import scipy
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from data_loader import TestData
from data_manager import *
from model import embed_net
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',
                    help='dataset name: regdb or sysu or llcm]')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--resume', '-r', default='sysu_djan_p4_n24_lr_0.1_seed_0_best.t', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='result/DJANNet/save_model_2/',
                    type=str, help='model save path')
parser.add_argument('--log_path', default='result/DJANNet/log/',
                    type=str, help='log save path')
parser.add_argument(
    '--vis_log_path', default='result/DJANNet/log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=2, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str,
                    metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=0, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='2', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str,
                    help='all or indoor for sysu')
parser.add_argument('--tvsearch', default=True,
                    help='retrive thermal to visible search on RegDB')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
need_evaluation = 1

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
pool_dim = 2048
print('==> Building model..')
if args.method == 'awg':
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
else:
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)

net.to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path

print('==> Loading data..')
# Data loading code

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize
])

end = time.time()


def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    gall_feat_fc_label = []
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_poolA, feat_fcA = net(
                input, input, input, input, test_mode[0])
            # feat_pool=feat_poolA[feat_poolA.size(0)//2:feat_poolA.size(0),:]
            # feat_fc=feat_fcA[feat_fcA.size(0)//2:feat_fcA.size(0),:]
            gall_feat_pool[ptr:ptr + batch_num,
            :] = feat_poolA.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num,
            :] = feat_fcA.detach().cpu().numpy()
            ptr = ptr + batch_num

            # ########
            if need_evaluation:
                gall_feat_fc_label.extend(label.detach().cpu().numpy())

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc, np.array(gall_feat_fc_label)


def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    query_feat_fc_label = []
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_poolA, feat_fcA = net(
                input, input, input, input, test_mode[1])
            query_feat_pool[ptr:ptr + batch_num,
            :] = feat_poolA.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num,
            :] = feat_fcA.detach().cpu().numpy()
            ptr = ptr + batch_num
            # ######
            if need_evaluation:
                query_feat_fc_label.extend(label.detach().cpu().numpy())

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc, np.array(query_feat_fc_label)


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

if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        print(model_path)
        # model_path = checkpoint_path + 'sysu_agw_p4_n8_lr_0.1_seed_0_best.t'
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path, map_location={
                'cuda:2': 'cuda:0', 'cuda:1': 'cuda:0'})
            # checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(
        data_path, mode=test_mode[0])
    gall_img, gall_label, gall_cam = process_gallery_sysu(
        data_path, mode=test_mode[1], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(
        len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(
        len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(
        args.img_w, args.img_h))
    query_loader = data.DataLoader(
        queryset, batch_size=args.test_batch, shuffle=False, num_workers=8)

    gallset = TestData(
        gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(
        gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


    query_feat_pool, query_feat_fc, query_feat_fc_label = extract_query_feat(        query_loader)
    gall_feat_pool, gall_feat_fc, gall_feat_fc_label = extract_gall_feat(        gall_loader)
    result = {'gallery_f': gall_feat_fc, 'gallery_label': gall_label, 'gallery_cam': gall_cam, 'query_f': query_feat_fc,
              'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat('result/DJANet/vis/sysu_result_{}_{}.mat'.format(test_mode[0], test_mode[1]), result)


elif dataset == 'regdb':
    test_trial = 1
    model_path = checkpoint_path + 'regdb_djan_p4_n6_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])

    # training set
    # trainset = RegDBData(data_path, test_trial, transform=transform_train)
    # # generate the idx of each person identity
    # color_pos, thermal_pos = GenIdx(
    #     trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(
        data_path, trial=test_trial, modal='visible')
    gall_img, gall_label = process_test_regdb(
        data_path, trial=test_trial, modal='thermal')

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(
        args.img_w, args.img_h))
    gall_loader = data.DataLoader(
        gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    nquery = len(query_label)
    ngall = len(gall_label)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(
        args.img_w, args.img_h))
    query_loader = data.DataLoader(
        queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc, label1 = extract_query_feat(
        query_loader)
    gall_feat_pool, gall_feat_fc, label2 = extract_gall_feat(gall_loader)
    result = {'gallery_f': gall_feat_fc, 'gallery_label': gall_label, 'gallery_cam': label2, 'query_f': query_feat_fc,
              'query_label': query_label, 'query_cam': label1}
    scipy.io.savemat('result/DJANNet/vis/regdb_result.mat', result)

elif dataset == 'llcm':
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2, query_label = extract_query_feat(query_loader)

    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    gall_feat1, gall_feat2, gall_label = extract_gall_feat(trial_gall_loader)

    result = {'gallery_f': gall_feat2, 'gallery_label': gall_label, 'gallery_cam': gall_cam, 'query_f': query_feat2,
              'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat('result/DJANNet/vis/llcm_result.mat', result)
