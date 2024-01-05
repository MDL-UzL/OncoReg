#!/usr/bin/env python

import torch
import sys
import os
import time
import argparse
from tqdm import trange,tqdm
from utils.vxmplusplus_utils import get_vxmpp_models,return_crops
from utils.thin_plate_spline import *
from utils.data_utils import get_files
from torch.utils.tensorboard import SummaryWriter

data_dir = 'data/'
dir_save = 'output/'

def main(args):
    
    #data_dir = "/home/heyer/storage/staff/wiebkeheyer/data/ThoraxCBCT/ThoraxCBCT_final_data/"
    #task = 'ThoraxCBCT'
    #data_dir = args.datadir
    task = args.task
    mode = 'Tr'
    do_MIND = False
    do_save = False  #  Write model and tensorboard logs?
    #dir_save = args.outdir
    #if do_save and not os.path.exists(dir_save):
    #    os.makedirs(dir_save)
    
    img_fixed_all, img_moving_all, kpts_fixed_all, kpts_moving_all, case_list, orig_shapes_all, mind_fixed_all, mind_moving_all, keypts_fixed_all, img_mov_unmasked, aff_mov_all = get_files(data_dir, task, mode, do_MIND)


    unet_model,heatmap,mesh = get_vxmpp_models()
    
    if do_save:  writer = SummaryWriter(log_dir=dir_save)

    for repeat in range(2):
        num_iterations = 4*4900
        optimizer = torch.optim.Adam(list(unet_model.parameters())+list(heatmap.parameters()),lr=0.001)#0.001
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,4*700,0.5)
        t0 = time.time()
        run_tre = torch.empty(0,1); run_tre_test = torch.empty(0,1); 
        run_loss = torch.zeros(num_iterations)

        with tqdm(total=num_iterations, file=sys.stdout) as pbar:


            for i in range(num_iterations):

                with torch.no_grad():
                    ii = torch.randperm(len(img_fixed_all))[0]#
                    
                    fixed_img = img_fixed_all[ii]

                    moving_img = img_moving_all[ii]

                    H,W,D = fixed_img.shape[-3:]

                    cf_fixed = kpts_fixed_all[ii]
                    cf_moving = kpts_moving_all[ii]

                    #halfres keypts
                    keypts_fix = torch.flip((cf_fixed-torch.tensor([H,W,D]))/torch.tensor([H,W,D]),(1,)).cuda()
                    keypts_mov = torch.flip((cf_moving-torch.tensor([H,W,D]))/torch.tensor([H,W,D]),(1,)).cuda()

                    #Affine augmentation of images *and* keypoints 
                    if(i%2==0):
                        A = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                        affine = F.affine_grid(A.unsqueeze(0),(1,1,H,W,D))
                        keypts_fix = torch.linalg.solve(torch.cat((A,torch.tensor([0,0,0,1]).cuda().view(1,-1)),0),\
                                        torch.cat((keypts_fix,torch.ones(keypts_fix.shape[0],1).cuda()),1).t()).t()[:,:3]
                        fixed_img = F.grid_sample(fixed_img.view(1,1,H,W,D).cuda(),affine)
                    else:
                        fixed_img = fixed_img.view(1,1,H,W,D).cuda()

                    if(i%2==1):
                        A = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
                        affine = F.affine_grid(A.unsqueeze(0),(1,1,H,W,D))
                        keypts_mov = torch.linalg.solve(torch.cat((A,torch.tensor([0,0,0,1]).cuda().view(1,-1)),0),\
                                        torch.cat((keypts_mov,torch.ones(keypts_mov.shape[0],1).cuda()),1).t()).t()[:,:3]
                        moving_img = F.grid_sample(moving_img.view(1,1,H,W,D).cuda(),affine)
                    else:
                        moving_img = moving_img.view(1,1,H,W,D).cuda()
                    disp_gt = keypts_mov-keypts_fix

                    scheduler.step()
                    optimizer.zero_grad()
                    idx = torch.randperm(keypts_fix.shape[0])[:1024]

                    with torch.cuda.amp.autocast():
                        #VoxelMorph requires some padding
                        input,x_start,y_start,z_start,x_end,y_end,z_end = return_crops(torch.cat((fixed_img,moving_img),1).cuda())
                        #input = F.interpolate(input,scale_factor=0.5,mode='trilinear')
                #end of no grad
                with torch.cuda.amp.autocast():

                    output = F.pad(F.interpolate(unet_model(input),scale_factor=2,mode='trilinear'),(z_start,(-z_end+D),y_start,(-y_end+W),x_start,(-x_end+H)))
                    #output = unet_model(input)
                    sample_xyz = keypts_fix[idx]#*torch.tensor([D,W,H]).cuda()/torch.tensor([320,256,320]).cuda()#keypts_all_fix[int(ii)][idx]#fix
                    #todo nearest vs bilinear
                    #sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
                    sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
                    #disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(512,-1,3,3,3))
                    disp_pred = heatmap(sampled.permute(2,1,0,3,4))


                    pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)
                    loss = (pred_xyz-disp_gt[idx]).mul(torch.tensor([D,W,H]).float().cuda()).pow(2).sum(-1).sqrt().mean()


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                run_loss[i] = loss.item()


                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
                if do_save:  writer.add_scalar("train_loss", loss, i)
               
        if(repeat==0):
            torch.save([heatmap.state_dict(),unet_model.state_dict(),run_loss],args.outdir + 'vxmpp_0.pth')
        else:
            torch.save([heatmap.state_dict(),unet_model.state_dict(),run_loss],args.outdir + 'vxmpp.pth')        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Training of VoxelMorph++')
    #parser.add_argument('-i',  '--datadir',   default='ThoraxCBCT', help="data folder containing imagesTr, masksTr, keypoints01Tr, keypoints02Tr")
    parser.add_argument('task',      default='ThoraxCBCT', help="task/dataset: ThoraxCBCT or OncoReg")
    #parser.add_argument('-o',  '--outdir',    default='models', help="output folder for trained model and tensorboard log")
    args = parser.parse_args()
    main(args)






