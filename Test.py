import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
# from scipy import misc
import cv2
from lib.Network import *
from utils.data_val import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='/data0/hcm/V6/snapshot/04_24_Weakly_SAM_COD_Scribble_10Points_Combined_weight2_masked2936/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']:
# for _data_name in ['CHAMELEON']:
    data_path = '/data0/hcm/dataset/COD/TestDataset/{}/'.format(_data_name)
    save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = Network_interFA_noSpade_noEdge_ODE_slot_channel4(channels=128) #can be different under diverse backbone
    # model = Network(channels=96)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.pth_path,map_location='cuda:0').items()})
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    # edge_root = '{}/Edge/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        # image, gt, edge, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        # edge = np.asarray(edge, np.float32)
        # edge /= (edge.max() + 1e-8)
        image = image.cuda()

        result = model(image)
        print('> {} - {}'.format(_data_name, name))

        # for j in range(64):

        res = F.upsample(result[4], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name,res*255)

        # res = F.upsample(result[0], size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # cv2.imwrite(save_path+'global_'+name,res*255)
        # res = F.upsample(result[1], size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # cv2.imwrite(save_path+'HH_'+name,res*255)
        # res = F.upsample(result[2], size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # cv2.imwrite(save_path+'HLLH_'+name,res*255)
        # res = F.upsample(result[3], size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # cv2.imwrite(save_path+'ll_'+name,res*255)
        # # # res_edge = F.upsample(result[8], size=gt.shape, mode='bilinear', align_corners=False)
        # # # res_edge = result[8]
        # # # res_edge = res_edge.sigmoid().data.cpu().numpy().squeeze()
        # # # res_edge = (res_edge - res_edge.min()) / (res_edge.max() - res_edge.min() + 1e-8)

        # # # misc.imsave(save_path+name, res)
        # # # If `mics` not works in your environment, please comment it and then use CV2
        # # cv2.imwrite(save_path+name,res*255)
        # # # cv2.imwrite(save_path+'edge_'+name,res_edge*255)
        # # # cv2.imwrite(save_path+'gt_'+name,res_edge_gt*255)
        # res_prior = F.upsample(result[0], size=gt.shape, mode='bilinear', align_corners=False)
        # res_prior = res_prior.sigmoid().data.cpu().numpy().squeeze()
        # res_prior = (res_prior - res_prior.min()) / (res_prior.max() - res_prior.min() + 1e-8)
        # cv2.imwrite(save_path+'prior_'+name,res_prior*255)
        #     res_LL = result[9][:,j,:,:]
        #     res_LL = res_LL.sigmoid().data.cpu().numpy().squeeze()
        #     res_LL = (res_LL - res_LL.min()) / (res_LL.max() - res_LL.min() + 1e-8)
        #     cv2.imwrite(save_path+'LL_'+str(j)+name,res_LL*255)
        #     res_LL = result[10][:,j,:,:]
        #     res_LL = res_LL.sigmoid().data.cpu().numpy().squeeze()
        #     res_LL = (res_LL - res_LL.min()) / (res_LL.max() - res_LL.min() + 1e-8)
        #     cv2.imwrite(save_path+'LH_'+str(j)+name,res_LL*255)
        #     res_LL = result[11][:,j,:,:]
        #     res_LL = res_LL.sigmoid().data.cpu().numpy().squeeze()
        #     res_LL = (res_LL - res_LL.min()) / (res_LL.max() - res_LL.min() + 1e-8)
        #     cv2.imwrite(save_path+'HL_'+str(j)+name,res_LL*255)
        #     res_LL = result[12][:,j,:,:]
        #     res_LL = res_LL.sigmoid().data.cpu().numpy().squeeze()
        #     res_LL = (res_LL - res_LL.min()) / (res_LL.max() - res_LL.min() + 1e-8)
        #     cv2.imwrite(save_path+'HH_'+str(j)+name,res_LL*255)
        # break

