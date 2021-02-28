import cv2
import numpy as np
import nibabel as nib
import torch
from torch import nn
import torch.nn.functional as F
import argparse
import os
from resnet_3d_cbam import *

from skimage.filters import threshold_otsu, gaussian

def parse_arg():
    parser = argparse.ArgumentParser(description='UCSD ImageClef2020')
    parser.add_argument('--img_pth',help='image dir',default='E:\Xing\TB2020\data\Test_data\CTR_TST_data',type=str)
    parser.add_argument('--msk1_pth', help='mask1 dir', default='E:\Xing\TB2020\data\Test_data\CTR_TST_masks1', type=str)
    parser.add_argument('--msk2_pth', help='mask2 dir', default='E:\Xing\TB2020\data\Test_data\CTR_TST_masks2', type=str)
    parser.add_argument('--img_id', help='image_id', default='CTR_TST_001.nii.gz', type=str)
    parser.add_argument('--base_dir', help='base dir', default='E:\Xing\TB2020\data\Test_data', type=str)

    parser.add_argument('--model_path', help='model dir', default= r'C:\Users\Xing\Projects\TB2020\train_log\Jun20_pw_resnet_cbam_bsmp_wfocal_fulldata\Sun21Jun2020-170404\save', type=str)
    parser.add_argument('--model_name', help='base dir', default=r'\best_model_auc.pth', type=str)
    parser.add_argument('--used_gpu', help='base dir', default='1,2', type=str)
    args = parser.parse_args()
    return args


class data_preprocess_config(object):
    # img_id = 'CTR_TST_001.nii.gz'
    # base_dir = r'E:\Xing\TB2020\data\Test_data'
    # img_pth = r'E:\Xing\TB2020\data\Test_data\CTR_TST_data'
    # msk1_pth = r'E:\Xing\TB2020\data\Test_data\CTR_TST_masks1'
    # msk2_pth = r'E:\Xing\TB2020\data\Test_data\CTR_TST_masks2'
    def __init__(self,args):
        self.img_id = args.img_id
        self.msk1_pth = args.msk1_pth
        self.msk2_pth = args.msk2_pth
        self.img_pth = args.img_pth
        self.base_dir = args.base_dir

class data_preprocess(object):
    def __init__(self,config):
        self.config = config

    def normalize_window(self, image_array, lower_sigma=2, upper_sigma=4, bg_thresh=None, bg_percentile=20, window='lung'):
        # select the fg pixels

        thresh_w = {
            'lung': [-600, 1500],
            'soft_tissue': [50, 350],
            'bone': [400, 1800]
        }

        image_array = image_array.astype(np.float)

        bg_thresh = threshold_otsu(image_array)

        # print('background threshold {}'.format(bg_thresh))

        image_array_fg = image_array[image_array > bg_thresh]

        # select 5 pct to 95 pct to perform robust normalization

        if window == 'normal':

            pct_5 = np.percentile(image_array_fg, 5)

            pct_95 = np.percentile(image_array_fg, 95)
        else:

            pct_5 = thresh_w[window][0]
            pct_95 = thresh_w[window][1]

        image_array_fg_robust = image_array_fg[(image_array_fg > pct_5) & (image_array_fg < pct_95)]

        std = np.std(image_array_fg_robust)

        mean = np.mean(image_array_fg_robust)

        # set (mean - lower_sigma * std) to 0, and (mean + upper_sigma * std) to 1

        a_min = mean - lower_sigma * std

        a_max = mean + upper_sigma * std

        # set bg pixels to a_min. Sometimes bg_threshold > a_min

        image_array[image_array <= bg_thresh] = a_min

        # clip

        image_array_clipped = np.clip(image_array, a_min=a_min, a_max=a_max)

        image_array_clipped = (image_array_clipped - a_min) / (a_max - a_min)

        return image_array_clipped


    def find_bound(self,msk, task='R'):
        # take upper part as R
        x, y = msk.shape
        img_binary = (msk > 0).astype(np.uint8)
        #     plt.imshow(img_binary)
        g = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, g)
        contours, hierarchy = cv2.findContours(img_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area = 0
        area_max = 0
        for i, c in enumerate(contours):
            #         area[i] = cv2.contourArea(c)
            #         print('the area is %d'%area[i])
            area = cv2.contourArea(c)
            if area_max < area:
                area_max = area
                c_max = c

        if len(contours) > 0:
            y_min = min(c_max[:, :, 1])[0]
            y_max = max(c_max[:, :, 1])[0]
        else:
            y_min = y
            y_max = 0  # this is a trick to avoid the interuptions.

        #     print('ymin and ymax =',y_min,y_max)

        if task == 'R':
            return y_max
        else:
            return y_min


    def msk_new(self, msk_s,msk_b,task = 'R'):
        x,y,z = msk_b.shape
        new_msk = np.zeros([x,y,z])
        for i in range(z):
            bound = self.find_bound(msk_s[:,:,i],task = task)
    #         print(i,bound)
            if task == 'R':
                new_msk[:bound,:,i] = msk_b[:bound,:,i]
            else:
                new_msk[bound:,:,i] = msk_b[bound:,:,i]
        return new_msk


    def img_crop_3d(self, image, msk, margin=2, ds=4, multi=False):
        if multi:
            image = image * msk
            print('image multiplied with msk')

        msk_z = np.sum(msk, axis=2)
        msk_y = np.sum(msk, axis=0)

        img_binary = (msk_z > 0).astype(np.uint8)
        g = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, g)
        contours, hierarchy = cv2.findContours(img_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area = 0
        area_max = 0
        minmax = []
        for i, c in enumerate(contours):
            #         area[i] = cv2.contourArea(c)
            #         print('the area is %d'%area[i])
            area = cv2.contourArea(c)
            #         if area_max < area:
            #             area_max = area
            #             c_max = c
            x_min = min(c[:, :, 0])[0] - margin
            x_max = max(c[:, :, 0])[0] + margin
            y_min = min(c[:, :, 1])[0] - margin
            y_max = max(c[:, :, 1])[0] + margin

            if area > 10:
                minmax.append([x_min, x_max, y_min, y_max])
        #     print(minmax)

        x_min = min(np.array(minmax)[:, 0])
        x_max = max(np.array(minmax)[:, 1])
        y_min = min(np.array(minmax)[:, 2])
        y_max = max(np.array(minmax)[:, 3])

        #     x_min = min(c_max[:,:,0])- margin
        #     x_max = max(c_max[:,:,0])+ margin
        #     y_min = min(c_max[:,:,1])- margin
        #     y_max = max(c_max[:,:,1])+ margin

        msk_y = np.sum(msk, axis=0)

        img_binary = (msk_y > 0).astype(np.uint8)
        g = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, g)
        contours, hierarchy = cv2.findContours(img_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area = 0
        area_max = 0
        for i, c in enumerate(contours):
            #         area[i] = cv2.contourArea(c)
            #         print('the area is %d'%area[i])
            area = cv2.contourArea(c)
            if area_max < area:
                area_max = area
                c_max = c

        z_min = min(c_max[:, :, 0])[0] + margin * ds
        z_max = max(c_max[:, :, 0])[0] - margin * ds

        print(x_min, x_max, y_min, y_max, z_min, z_max)

        #     img_crop = image[y_min[0]:y_max[0], x_min[0]:x_max[0],z_min[0]:z_max[0]]
        img_crop = image[y_min:y_max, x_min:x_max, z_min:z_max]

        return img_crop

    def __call__(self, *args, **kwargs):

        img = nib.load(os.path.join(self.config.img_pth, self.config.img_id))
        msk = nib.load(os.path.join(self.config.msk1_pth, self.config.img_id))
        mskw = nib.load(os.path.join(self.config.msk2_pth, self.config.img_id))

        img_affine = img.affine

        sl = img.header['pixdim'][3]
        x, y, z = img.shape
        center_sl = z // 2

        msk_1 = np.zeros(msk.shape)
        msk_2 = np.zeros(msk.shape)
        msk_w = np.zeros(msk.shape)

        img_f = img.get_fdata()
        msk = msk.get_fdata()
        mskw = mskw.get_fdata()
        msk_1[msk == 1] = 1
        msk_2[msk == 2] = 1

        msk_1 = self.msk_new(msk_1, mskw, task='R')  # upper part as R as 1
        msk_2 = self.msk_new(msk_2, mskw, task='L')  # lower part as L as 2

        if sl > 2.5:
            ds = 1
        elif sl > 1.25:
            ds = 2
        else:
            ds = 4
        print(x, y, z, sl, ds)

        img_d = self.normalize_window(img_f, window='normal')
        img_l = self.normalize_window(img_f, window='lung')
        img_s = self.normalize_window(img_f, window='soft_tissue')
        img_b = self.normalize_window(img_f, window='bone')

        meta_data = {}

        msk_1d = img_d * msk_1
        msk_2d = img_d * msk_2

        img_1d = self.img_crop_3d(img_d, msk_1d, ds=ds, multi=True)
        img_1l = self.img_crop_3d(img_l, msk_1d, ds=ds, multi=True)
        img_1s = self.img_crop_3d(img_s, msk_1d, ds=ds, multi=True)

        x, y, z = img_1d.shape

        meta_data['Right'] = {}
        meta_data['Left'] = {}

        meta_data['Right'] = {'data': img_1d, 'data_l': img_1l, 'data_s': img_1s, 'mask': msk_1d, 'len': z, 'ds': ds}

        img_2d = self.img_crop_3d(img_d, msk_2d, ds=ds, multi=True)
        img_2l = self.img_crop_3d(img_l, msk_2d, ds=ds, multi=True)
        img_2s = self.img_crop_3d(img_s, msk_2d, ds=ds, multi=True)

        x, y, z = img_2d.shape

        meta_data['Left'] = {'data': img_2d, 'data_l': img_2l, 'data_s': img_2s, 'mask': msk_2d, 'len': z, 'ds': ds}

        return meta_data


class model_infer_config(object):
    def __init__(self,args):
        self.model_type = 'cbam_bsmp_wfocal_fulldata'
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.used_gpu = args.used_gpu
        self.num_classes = 3

class model_infer(object):
    def __init__(self,config):
        self.config = config
        self.gpu = config.used_gpu
        self.num_classes = config.num_classes
        self.len = 64
        self.imsz = 256
        self.inputs_val = torch.rand(1, 4, self.len, self.imsz, self.imsz).cuda()
        self.get_model()
        # self.transform = transforms.ToTensor()
        self.transform = None

    def interp(self, img, con_len):
        img_t = torch.tensor(img)
        img_t = img_t.unsqueeze(0).unsqueeze(0)

        img_r = F.interpolate(img_t, [self.imsz, self.imsz, con_len])
        img_r = np.array(img_r).squeeze()
        return img_r

    def get_data(self,data):
        # self.input_val = torch.autograd.Variable(torch.rand(1,4,64,256,256)).cuda()
        # self.inputs_val = torch.rand(1, 4, 64, 256, 256).cuda()
        img = data['data']
        img_l = data['data_l']
        img_s = data['data_s']

        img = self.interp(img, self.len)
        img_l = self.interp(img_l, self.len)
        img_s = self.interp(img_s, self.len)

        img_c = np.array([img, img_l, img_s, img])

        image = torch.tensor(img_c.transpose(0, 3, 1, 2))

        if self.transform:
            image = self.transform(image)

        self.inputs_val = image.unsqueeze(dim=0)


    def get_model(self):

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        self.model = resnet34(sample_size=self.imsz, sample_duration=self.len, num_classes=self.num_classes)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model).cuda()

        checkpoint = torch.load(self.config.model_path + self.config.model_name)
        # print(key, checkpoint['epoch'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()
        self.model.eval()

    def __call__(self, *args, **kwargs):
        outputs_val = self.model(self.inputs_val.float().cuda())
        outputs_val = torch.sigmoid(outputs_val)
        return outputs_val.detach().cpu().numpy()


if __name__ == "__main__":

    args = parse_arg()

    img_config = data_preprocess_config(args=args)

    model_config = model_infer_config(args = args)

    meta_data = data_preprocess(config=img_config)()

    infered_model = model_infer(config=model_config)

    for key in meta_data.keys():
        infered_model.get_data(meta_data[key])
        results = infered_model()
        meta_data[key]['predict'] = results
        print(key,results)