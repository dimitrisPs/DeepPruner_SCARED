
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import skimage
import skimage.transform
import numpy as np
from dataloader import scared_collector as ls
from dataloader import scared_loader as DA
from models.deeppruner import DeepPruner

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from loss_evaluation import loss_evaluation
import matplotlib.pyplot as plt
import logging
from setup_logging import setup_logging
from pathlib import Path


from utils import Params

parser = argparse.ArgumentParser(description='DeepPruner')

parser.add_argument('--config', required=True, help='path to config file.')

args = parser.parse_args()
config = Params(args.config)

args.cuda = config.cuda and torch.cuda.is_available()
torch.manual_seed(config.seed)
if args.cuda:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True


setup_logging(Path(args.config).parent / (Path(args.config).parent.name + '.log'))
logger = logging.getLogger()
logger.disabled = True
if hasattr(config, 'chery'):
    print(config.chery)
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.datacollector(
        config.datapath, dense=config.dense, chery_pick=config.chery)
else:
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.datacollector(
        config.datapath, dense=config.dense)
print(len(all_left_img), len(all_left_img), len(all_left_disp))
print(len(test_left_img), len(test_left_img), len(test_left_disp))


TrainImgLoader = torch.utils.data.DataLoader(
         DA.SCAREDLoader(all_left_img, all_right_img, all_left_disp, training=True, disp_scaling=128.0),
         batch_size=config.batch_train, shuffle=True, num_workers=config.num_workers*2, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.SCAREDLoader(test_left_img, test_right_img, test_left_disp, training=False, disp_scaling=128.0),
         batch_size=config.batch_eval, shuffle=False, num_workers=config.num_workers, drop_last=False)

model = DeepPruner()
writer = SummaryWriter(str(Path(args.config).parent))
model = nn.DataParallel(model)

if args.cuda:
    model.cuda()

if config.loadmodel !='':
    state_dict = torch.load(config.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=True)


optimizer = optim.Adam(model.parameters(), lr=config.optim_learning_rate, betas=(0.9, 0.999), weight_decay=config.optim_lr_decay)


def train(imgL, imgR, disp_L, iteration, epoch):


    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    mask = (disp_true > 0)
    mask.detach_()


    optimizer.zero_grad()
    result = model(imgL,imgR)
    loss, _ = loss_evaluation(result, disp_true, mask, config.cost_aggregator_scale)

    loss.backward()
    optimizer.step()

    return loss.item()

    
    
def test(imgL,imgR,disp_L,iteration):

        model.eval()
        with torch.no_grad():

            if args.cuda:
                imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

            mask = (disp_true > 0)
            mask.detach_()
            
            optimizer.zero_grad()
            
            result = model(imgL,imgR)
            loss, output_disparity = loss_evaluation(result, disp_true, mask, config.cost_aggregator_scale)

            #computing 3-px error: (source psmnet)#
            true_disp = disp_true.data.cpu()
            disp_true = true_disp
            pred_disp = output_disparity.data.cpu()

            index = np.argwhere(true_disp>0)
            disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
            correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
            torch.cuda.empty_cache()             
            
            error = 1-(float(torch.sum(correct))/float(len(index[0])))
                
        return loss, error

def main():

    for epoch in range(0, config.epochs):
        # print (epoch)
        total_train_loss = 0
        total_test_loss = 0
        total_error = 0


    
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss, error = test(imgL,imgR,disp_L,batch_idx)
            total_test_loss += test_loss.item()
            total_error += error
        

        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            loss = train(imgL_crop,imgR_crop,disp_crop_L,batch_idx,epoch)
            total_train_loss += loss
        

        writer.add_scalar('Loss/train',total_train_loss/len(TrainImgLoader),epoch)
        writer.add_scalar('Loss/evaluation',total_test_loss/len(TestImgLoader),epoch)
        writer.add_scalar("error",total_error/len(TestImgLoader),epoch)
        writer.add_scalar("error",total_error/len(TestImgLoader),epoch)
        print (epoch, total_train_loss/len(TrainImgLoader), total_test_loss/len(TestImgLoader))
        # SAVE
        if epoch%config.save_summary_steps==0:
            savefilename = Path(args.config).parent / 'weights' / ('finetune_'+str(epoch)+'.tar')
            savefilename.parent.mkdir(exist_ok=True, parents=True)
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss,
                    'test_loss': total_test_loss,
                }, savefilename)


if __name__ == '__main__':
    main()
