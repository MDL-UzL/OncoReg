import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from tqdm import trange,tqdm
import sys
import os
import json
sys.path.insert(0,'corrfield/')
from foerstner import foerstner_kpts 
from vxmplusplus_utils import MINDSSC

def read_image_folder(base_img,base_mask,do_MIND = True):
    

    img_list = [i.split('Ts/')[1] for i in sorted(glob(base_img+'/*.nii.gz'))]
    mask_list = [i.split('Ts/')[1] for i in sorted(glob(base_mask+'/*.nii.gz'))]
    if(img_list != mask_list):
        raise Exception("Masks and images do not correctly match. Please check folder content.")

    img_list1 = [i.replace('_0000.nii.gz','').replace('_0001.nii.gz','') for i in img_list]
    #if(len(img_list1) != 2*len(set(img_list1))):
        #raise Exception("Not all images seem to have both _0000.nii.gz and _0001.nii.gz")
    case_list = sorted(list(set(img_list1)))
    img_insp_all = []
    img_exp_all = []
    keypts_insp_all = []
    mind_insp_all = []
    mind_exp_all = []
    orig_shapes_all = []
    print('Loading '+str(len(case_list))+' scan pairs')

    for ii in trange(len(case_list)):
        i = int(ii)

        img_exp = torch.from_numpy(nib.load(base_img+'/'+case_list[ii]+'_0000.nii.gz').get_fdata()).float()
        mask_exp = torch.from_numpy(nib.load(base_mask+'/'+case_list[ii]+'_0000.nii.gz').get_fdata()).float()
        masked_exp = F.interpolate(((img_exp+1024)*mask_exp).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

        img_insp = torch.from_numpy(nib.load(base_img+'/'+case_list[ii]+'_0001.nii.gz').get_fdata()).float()
        mask_insp = torch.from_numpy(nib.load(base_mask+'/'+case_list[ii]+'_0001.nii.gz').get_fdata()).float()
        
        kpts_fix = foerstner_kpts(img_insp.unsqueeze(0).unsqueeze(0).cuda(), mask_insp.unsqueeze(0).unsqueeze(0).cuda(), 1.4, 3).cpu()
        keypts_insp_all.append(kpts_fix)
            
        masked_insp = F.interpolate(((img_insp+1024)*mask_insp).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

        shape1 = mask_insp.shape
        

        img_exp_all.append(masked_exp)
        orig_shapes_all.append(shape1)

        img_insp_all.append(masked_insp)

        if(do_MIND):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    mind_insp = F.avg_pool3d(mask_insp.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_insp.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()
                    mind_exp = F.avg_pool3d(mask_exp.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_exp.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()

            mind_insp_all.append(mind_insp)
            mind_exp_all.append(mind_exp)
            del mind_insp
            del mind_exp
            
    return img_insp_all,img_exp_all,keypts_insp_all, mind_insp_all,mind_exp_all,orig_shapes_all,case_list


def get_files(data_dir, kpt_dir, task, mode, do_MIND):

    if task == "ThoraxCBCT":
        data_json = os.path.join(data_dir, task + "_dataset.json")

        with open(data_json) as file:
            data = json.load(file)

        if mode == 'Tr':
            mode1 = 'training_paired_images'
        elif mode == 'Val':
            mode = 'Tr'
            mode1 = 'registration_val'
        elif mode == 'Ts':
            mode1 = 'registration_test'

        img_fixed_all = []
        img_moving_all = []
        kpts_all = []
        orig_shapes_all = []
        mind_fixed_all = []
        mind_moving_all = []
        case_list = []
        keypts_fixed_all = []
        img_mov_unmasked = []
        #num_cases = len(data["training_paired_images"])
        for pair in data[mode1]:
            nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
            nam_moving = os.path.basename(pair["moving"]).split(".")[0]
            #if nam_fixed.split('.nii.gz')[0].split('_')[2]=='0001':
            #    kpts_dir = kpt_dir + 'keypoints01Tr'
            #else:
            #    kpts_dir = kpt_dir + 'keypoints02Tr'
     
            case_list.append(nam_fixed)

            img_fixed = torch.from_numpy(nib.load(os.path.join(data_dir, "images" + mode, nam_fixed + ".nii.gz")).get_fdata()).float()
            img_moving = torch.from_numpy(nib.load(os.path.join(data_dir, "images" + mode, nam_moving + ".nii.gz")).get_fdata()).float()
            #"fixed_label": os.path.join('/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/additional_data/TSv2/ml_13' , nam_fixed + ".nii.gz"),
            #"moving_label": os.path.join('/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/additional_data/TSv2/ml_13' , nam_moving + ".nii.gz"),
            label_fixed = torch.from_numpy(nib.load(os.path.join(data_dir, 'masks' + mode, nam_fixed + ".nii.gz")).get_fdata()).float()
            label_moving = torch.from_numpy(nib.load(os.path.join(data_dir, 'masks' + mode, nam_moving + ".nii.gz")).get_fdata()).float()
            #kpts_fixed = torch.from_numpy(nib.load(os.path.join(kpts_dir , nam_fixed + ".csv")).get_fdata()).float()
            #kpts_moving = torch.from_numpy(nib.load(os.path.join(kpts_dir , nam_moving + ".csv")).get_fdata()).float()
            kpts = torch.from_numpy(np.loadtxt(kpt_dir +mode+'/'+ nam_fixed + ".csv",delimiter=',')).float()


            masked_fixed = F.interpolate(((img_fixed+1024)*label_fixed).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()
            masked_moving = F.interpolate(((img_moving+1024)*label_moving).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

            shape = label_fixed.shape

            kpts_fix = foerstner_kpts(img_fixed.unsqueeze(0).unsqueeze(0).cuda(), label_fixed.unsqueeze(0).unsqueeze(0).cuda(), 1.4, 3).cpu()
            keypts_fixed_all.append(kpts_fix)
            

            img_fixed_all.append(masked_fixed)
            img_moving_all.append(masked_moving)
            kpts_all.append(kpts)
            orig_shapes_all.append(shape)
            img_mov_unmasked.append(img_moving)

            if(do_MIND):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        mind_fixed = F.avg_pool3d(label_fixed.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_fixed.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()
                        mind_moving = F.avg_pool3d(label_moving.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_moving.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()

                mind_fixed_all.append(mind_fixed)
                mind_moving_all.append(mind_moving)
                del mind_fixed
                del mind_moving
                     
            
    else:
        raise ValueError(f"Task {task} undefined!")
    
    return img_fixed_all, img_moving_all, kpts_all, case_list, orig_shapes_all, mind_fixed_all, mind_moving_all, keypts_fixed_all, img_mov_unmasked
