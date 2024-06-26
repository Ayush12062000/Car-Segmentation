import torch
import torch.nn as nn
import os
import numpy as np
import kornia
from kornia.augmentation import *
import cv2
from torch.utils.data import Dataset
import torchmetrics
import monai
import gc

import warnings
warnings.filterwarnings("ignore")

# Pre-processing
class PreProcess(nn.Module):
    '''
    Class to convert numpy array into torch tensor
    '''
    
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()  #disable gradients for efficiency
    def forward(self, x: np.array) -> torch.tensor:
        temp: np.ndarray = np.asarray(x) # HxWxC
        out: torch.tensor = kornia.image_to_tensor(temp, keepdim=True)  # CxHxW
        
        return out.float()


# Dataset Class
class SegmentationDataset(Dataset):
    
    def __init__(self, dirPath= r'../data', imageDir='images', masksDir='masks', img_size=512):
        self.imgDirPath = os.path.join(dirPath, imageDir)
        self.maskDirPath = os.path.join(dirPath, masksDir)
        self.img_size = img_size
        self.nameImgFile = sorted(os.listdir(self.imgDirPath))
        self.nameMaskFile = sorted(os.listdir(self.maskDirPath))
        self.preprocess = PreProcess()
    
    def __len__(self):
        return len(self.nameImgFile)
    
    def __getitem__(self, index):
        imgPath = os.path.join(self.imgDirPath, self.nameImgFile[index])
        maskPath = os.path.join(self.maskDirPath, self.nameMaskFile[index])
        
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Min-max scaling
        imin, imax = resized_img.min(), resized_img.max()
        resized_img = (resized_img-imin)/(imax-imin)
        
        img = self.preprocess(resized_img) 
        
        mask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
        resized_mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        mask = self.preprocess(resized_mask)
        
        return img, mask

# Get Metric
def getScore(model, data):
    
    dice_score = []
    iou_list = []
    for idx in range(len(data)):
        imgs, masks, _ = data[idx]

        with torch.no_grad():
            pred = model(imgs.unsqueeze(0).cuda())
            pred = torch.sigmoid(pred).squeeze().cpu()
            
        dice = monai.metrics.compute_meandice(pred, masks.squeeze())
        iou = torchmetrics.functional.jaccard_index(pred.squeeze(), masks.int().squeeze(), num_classes=2)

        dice_score.append(dice.item())
        iou_list.append(iou.item())
        del pred
        gc.collect()

    print(f'Mean Dice: {round(np.mean(dice_score),3)}')
    print(f'Mean IOU: {round(np.mean(iou_list),3)}')

# Checkpoint utils
def save_checkpoint(state, filename="../Model Checkpoints/car_segmentation_single_channel_checkpoint.pth.tar"):
    
    '''
    e.g:
    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }

    save_checkpoint(checkpoint)
    '''
    
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    
    '''
    e.g: load_checkpoint(torch.load("../Model Checkpoints/car_segmentation_single_channel_checkpoint.pth.tar"), model)
    
    checkpoint: path of model checkpoint
    model: model Object
    '''
    
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])