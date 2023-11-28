
import torch
import sys
from vxmplusplus_utils import get_vxmpp_models,return_crops
from thin_plate_spline import *
from tqdm import trange
from vxmplusplus_utils import adam_mind
import argparse
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from glob import glob
import os

from thin_plate_spline import thin_plate_dense
from scipy.ndimage.interpolation import zoom as zoom
from scipy.ndimage.interpolation import map_coordinates
from data_utils import read_image_folder, get_files

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='6'

def get_warped_pair(_cf,_img_moving):
    H,W,D = _img_moving.shape[-3:]
    
    kpts_fixed = torch.flip((_cf[:,:3]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
    kpts_moving = torch.flip((_cf[:,3:]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))


    with torch.no_grad():
        dense_flow = thin_plate_dense(kpts_fixed.unsqueeze(0).cuda(), (kpts_moving-kpts_fixed).unsqueeze(0).cuda(), (H, W, D), 4, 0.01).cpu()
    warped_img = F.grid_sample(_img_moving.view(1,1,H,W,D).cpu(),dense_flow+F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D))).squeeze()

        
    return warped_img,dense_flow

def main(args):
    
    data_dir = "/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/ThoraxCBCT_final_data/"
    kpt_dir = '/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/L2R_baselines/convexAdam/keypoints'
    task = 'ThoraxCBCT'
    mode = 'Ts'
    model = '/home/heyer/storage/staff/wiebkeheyer/oncoreg/OncoReg/vxmpp/models/vxmpp_11_24/vxmpp_11_24.pth'
    outfile = '/home/heyer/storage/staff/wiebkeheyer/oncoreg/OncoReg/vxmpp/results/vxmpp_11_24/predictions.pth'
    outfolder = '/home/heyer/storage/staff/wiebkeheyer/oncoreg/OncoReg/vxmpp/results/vxmpp_11_24'
    do_MIND = True
    #img_insp_all,img_exp_all,keypts_insp_all,mind_insp_all,mind_exp_all,orig_shapes_all,case_list = read_image_folder(args.imgfolder,args.maskfolder,do_MIND=True)
    img_fixed_all, img_moving_all, kpts_all, case_list, orig_shapes_all, mind_fixed_all, mind_moving_all, keypts_fixed_all, img_mov_unmasked = get_files(data_dir, kpt_dir, task, mode, do_MIND)

    unet_model,heatmap,mesh = get_vxmpp_models()

    state_dicts = torch.load(model)
    unet_model.load_state_dict(state_dicts[1])
    heatmap.load_state_dict(state_dicts[0])

    print('inference for validation scans with TRE computation ')

    predictions = []
    
    for case in trange(len(case_list)):
        #ii = int(case_list[case].split('case_')[1])

        ##MASKED INPUT IMAGES ARE HALF-RESOLUTION
        #dataset = datasets[ii]
        with torch.no_grad():
            fixed_img = img_fixed_all[case]
            moving_img = img_moving_all[case]
            keypts_fix = keypts_fixed_all[case].squeeze().cuda()


            H,W,D = fixed_img.shape[-3:]

            fixed_img = fixed_img.view(1,1,H,W,D).cuda()
            moving_img = moving_img.view(1,1,H,W,D).cuda()

            with torch.cuda.amp.autocast():
                #VoxelMorph requires some padding
                input,x_start,y_start,z_start,x_end,y_end,z_end = return_crops(torch.cat((fixed_img,moving_img),1).cuda())
                output = F.pad(F.interpolate(unet_model(input),scale_factor=2),(z_start,(-z_end+D),y_start,(-y_end+W),x_start,(-x_end+H)))
                disp_est = torch.zeros_like(keypts_fix)
                for idx in torch.split(torch.arange(len(keypts_fix)),1024):
                    sample_xyz = keypts_fix[idx]
                    sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
                    disp_pred = heatmap(sampled.permute(2,1,0,3,4))
                    disp_est[idx] = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)


        ##NOW EVERYTHING FULL-RESOLUTION
        H,W,D = orig_shapes_all[case]#.shape[-3:]

        fixed_mind = mind_fixed_all[case].view(1,-1,H//2,W//2,D//2).cuda()
        moving_mind = mind_moving_all[case].view(1,-1,H//2,W//2,D//2).cuda()

        pred_xyz,disp_smooth,dense_flow = adam_mind(keypts_fix,disp_est,fixed_mind,moving_mind,H,W,D)
        predictions.append(pred_xyz.cpu()+keypts_fix.cpu())
        ##EVALUATION WITH MANUAL LANDMARKS PROVIDED WITH LUNG-250M-4B
        #tre0 = (lms_validation[str(ii)][:,:3]-lms_validation[str(ii)][:,3:]).pow(2).sum(-1).sqrt()
        #lms1 = torch.flip((lms_validation[str(ii)][:,:3]-torch.tensor([H/2,W/2,D/2]))/torch.tensor([H/2,W/2,D/2]),(1,))
        #lms_disp = F.grid_sample(disp_smooth.cpu(),lms1.view(1,-1,1,1,3)).squeeze().t()

        #tre2 = (lms_validation[str(ii)][:,:3]+lms_disp-lms_validation[str(ii)][:,3:]).pow(2).sum(-1).sqrt()
        #print(dataset+'-TRE init: '+str('%0.3f'%tre0.mean())+'mm; net+adam: '+str('%0.3f'%tre2.mean())+'mm;')


    torch.save({'keypts_mov_predict':predictions,'case_list':case_list,'keypts_fix':keypts_fixed_all},outfile)
    if(outfolder is not None):
        for i in range(len(case_list)):
            case = case_list[i]
            output_path = outfolder+'/'+case
            H,W,D = orig_shapes_all[i]
            kpts_fix = torch.flip(keypts_fixed_all[i].squeeze(),(1,))*torch.tensor([H/2,W/2,D/2])+torch.tensor([H/2,W/2,D/2])
            kpts_moved = torch.flip(predictions[i].squeeze(),(1,))*torch.tensor([H/2,W/2,D/2])+torch.tensor([H/2,W/2,D/2])
            np.savetxt('{}.csv'.format(output_path), torch.cat([kpts_fix, kpts_moved], dim=1).cpu().numpy(), delimiter=",", fmt='%.3f')

            #img_mov = torch.from_numpy(nib.load('/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/ThoraxCBCT_final_data/imagesTr/ThoraxCBCT_0011_0000.nii.gz').get_fdata()).float()
            img_mov = img_mov_unmasked[i]
            aff_mov = nib.load('/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/ThoraxCBCT_final_data/imagesTr/ThoraxCBCT_0011_0000.nii.gz').affine

            cf = torch.from_numpy(np.loadtxt(outfolder+'/'+case+'.csv',delimiter=',')).float()
            kpts_fixed = torch.flip((cf[:,:3]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
            kpts_moving = torch.flip((cf[:,3:]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
            with torch.no_grad():
                dense_flow = thin_plate_dense(kpts_fixed.unsqueeze(0).cuda(), (kpts_moving-kpts_fixed).unsqueeze(0).cuda(), (H, W, D), 4, 0.001)
            warped_img = F.grid_sample(img_mov.view(1,1,H,W,D),dense_flow.cpu()+F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D))).squeeze()
            warped = nib.Nifti1Image(warped_img.numpy(), aff_mov)  
            nib.save(warped, '/home/heyer/storage/staff/wiebkeheyer/oncoreg/OncoReg/vxmpp/results/vxmpp_11_24/warped_' + case + '.nii.gz')
            
            dense_flow = dense_flow.cpu().flip(4).permute(0, 4, 1, 2, 3) * torch.tensor( [H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1) / 2
            grid_sp = 1
            disp_lr = F.interpolate(dense_flow, size=(H // grid_sp, W // grid_sp, D // grid_sp), mode='trilinear',
                                                align_corners=False)
            disp_lr = disp_lr.permute(0,2,3,4,1)
            disp_tmp = disp_lr[0].permute(3,0,1,2).numpy()
            disp_lr = disp_lr[0].numpy()
            displacement_field = nib.Nifti1Image(disp_lr, aff_mov)
            nib.save(displacement_field, '/home/heyer/storage/staff/wiebkeheyer/oncoreg/OncoReg/vxmpp/results/vxmpp_11_24/disp_field_' + case + '.nii.gz')
            


    '''for i in range(len(case_list)):
        case = case_list[i]
        img_mov = torch.from_numpy(nib.load(args.imgfolder+'/'+case+'_0000.nii.gz').get_fdata()).float()
        
        cf = torch.from_numpy(np.loadtxt(outfolder+'/'+case+'.csv',delimiter=',')).float()
        img_warped,dense_flow = get_warped_pair(cf,img_mov)
        print(img_mov.shape)
        H,W,D = img_mov.shape
        #H,W,D = moving_img.shape[-3:]
        dense_flow = dense_flow.flip(4).permute(0, 4, 1, 2, 3) * torch.tensor( [H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1) / 2
        grid_sp = 1
        disp_lr = F.interpolate(dense_flow, size=(H // grid_sp, W // grid_sp, D // grid_sp), mode='trilinear',
                                            align_corners=False)
        disp_lr = disp_lr.permute(0,2,3,4,1)
        disp_tmp = disp_lr[0].permute(3,0,1,2).numpy()
        disp_lr = disp_lr[0].numpy()
        A = nib.load(args.imgfolder+'/'+case+'_0000.nii.gz').affine
        displacement_field = nib.Nifti1Image(disp_lr, None)
        nib.save(displacement_field, '/share/data_supergrover3/heyer/n_ThoraxCBCT/VoxelMorphPlusPlus/results_aufTrainingsdatenCheck/' + case + '_disp.nii.gz')
        
        identity = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
        
        moving_warped = map_coordinates(img_mov, identity + disp_tmp, order=0)
        moving_warped = nib.Nifti1Image(moving_warped, A)
        nib.save(moving_warped,'/share/data_supergrover3/heyer/n_ThoraxCBCT/VoxelMorphPlusPlus/results_aufTrainingsdatenCheck/' + case + '_0000_warped.nii.gz')

        cf = torch.from_numpy(np.loadtxt('/home/heyer/storage/staff/wiebkeheyer/tmp/kpts_convAdam/keypoints.csv',delimiter=',')).float()
        kpts_fixed = torch.flip((cf[:,:3]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
        kpts_moving = torch.flip((cf[:,3:]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))'''


        
        
        
        

if __name__ == "__main__":

    #parser = argparse.ArgumentParser(description = 'inference of VoxelMorph++ on Lung250M-4B')

    #parser.add_argument('-M',  '--model',        default='registration_models/voxelmorphplusplus.pth', help="model file (pth)")
    #parser.add_argument('-m',  '--maskfolder',   default='masksTs', help="mask folder containing (/case_???_{1,2}.nii.gz)")
    #parser.add_argument('-I',  '--imgfolder',    default='imagesTs', help="image folder containing (/case_???_{1,2}.nii.gz)")
    #parser.add_argument('-O',  '--outfile',      default='predictions.pth', help="output file for keypoint displacement predictions")
    #parser.add_argument('-o',  '--outfolder',    default=None, help="output folder for individual keypoint displacement predictions")

    #args = parser.parse_args()
    #print(args)
    args = None
    main(args)






