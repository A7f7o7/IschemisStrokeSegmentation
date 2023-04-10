import torch
import torchvision
import matplotlib.pyplot as plt
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

import itk
import re

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

def resample(img):
    metadataIMG=itk.dict_from_image(img)


    pixdims=[float(metadataIMG['spacing'][0]),float(metadataIMG['spacing'][1]),float(metadataIMG['spacing'][2])]

    dims=[int(metadataIMG['size'][0]),float(metadataIMG['size'][1]),float(metadataIMG['size'][2])]
    
    new_spacing=[2.5, 2.5, 2.5]

    interpolator = itk.LinearInterpolateImageFunction.New(img)

    resampler=itk.ResampleImageFilter.New(img)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(itk.size(img))
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetSize([int((pixdims[0]/2)*dims[0]),int((pixdims[1]/2)*dims[1]),int((pixdims[2]/2)*dims[2])])
    #resampler.SetSize([100,100,100])
    resampler.Update()
    resampled=resampler.GetOutput()
    return resampled

class MyDataset(Dataset):
    def __init__(self, img_dir, masks_dir, transform=None):
        self.img_dir = img_dir
        self.masks_dir = masks_dir
        #self.imgs_names = os.listdir(img_dir)
        #attempt below to filter to only adc MRI modality
        #self.imgs_names = [f for f in os.listdir(img_dir) if re.match('*adc*', f)]
        self.imgs_names = []
        self.imgs_adc = []
        self.imgs_dwi = []
        self.imgs_flair = []
        for file_name in os.listdir(img_dir):
            if 'adc' in file_name:
                self.imgs_adc.append(file_name)
            elif 'dwi' in file_name:
                self.imgs_dwi.append(file_name)
            elif 'flair' in file_name:
                self.imgs_flair.append(file_name)
            else:
                continue
            self.imgs_names.append(file_name)

        
        self.masks_names = os.listdir(masks_dir)
        self.imgs_len = len(self.imgs_adc)
        self.transform = transform

    def __len__(self):
        print("Total images length: ", len(self.imgs_names))
        return self.imgs_len

    def __getitem__(self, idx, normalize=True):
        
        adc_path = os.path.join(self.img_dir, self.imgs_adc[idx]).replace("\\","/")
        dwi_path = os.path.join(self.img_dir, self.imgs_dwi[idx]).replace("\\","/")
        flair_path = os.path.join(self.img_dir, self.imgs_flair[idx]).replace("\\","/")
        seg_path = os.path.join(self.masks_dir, self.masks_names[idx]).replace("\\","/")
        
        adc_image = itk.imread(adc_path)
        dwi_image = itk.imread(dwi_path)
        flair_image = itk.imread(flair_path)
        seg = itk.imread(seg_path)

        RES_adc_image = resample(adc_image)
        RES_dwi_image = resample(dwi_image)
        RES_flair_image = resample(flair_image)
        RESseg = resample(seg)

        Cropped_adc_RESImage =  itk.array_from_image(RES_adc_image).astype(np.float32)[:50, :100, :75]
        Cropped_dwi_RESImage =  itk.array_from_image(RES_dwi_image).astype(np.float32)[:50, :100, :75]
        Cropped_flair_RESImage =  itk.array_from_image(RES_flair_image).astype(np.float32)[:50, :100, :75]
        
        CroppedRESSeg = itk.array_from_image(RESseg).astype(np.int16)[:50, :100, :75]
        
        multimodal = np.zeros((3, 50, 100, 75), dtype=np.float32)
        
        multimodal[0, ...] = Cropped_adc_RESImage
        multimodal[1, ...] = Cropped_dwi_RESImage
        multimodal[2, ...] = Cropped_flair_RESImage
        
        if (normalize==True):
            mean = np.mean(multimodal)
            std = np.std(multimodal)
            normalized_multimodal = (multimodal - mean) / std
        
            mean = np.mean(CroppedRESSeg)
            std = np.std(CroppedRESSeg)
            normalized_CroppedRESSeg = (CroppedRESSeg - mean) / std
            
        #if self.transform:
            #print(image.shape)
            #for i in range(0,2):
                #pil_image = transforms.ToPILImage()(multimodal[i])
                #image = self.transform(pil_image)
        
            #print("This print is post-normalization")
        

        #return itk.array_from_image(RESimage).astype(np.float32)[:50, :100, :100], itk.array_from_image(RESseg).astype(np.int16)[:50, :100, :100]
            
            return normalized_multimodal, normalized_CroppedRESSeg
        else:
            return multimodal, CroppedRESSeg
        
    def __getITKObjects__(self, idx, resampling=True):
        img_path = os.path.join(self.img_dir, self.imgs_names[idx]).replace("\\","/")
        #seg_path = os.path.join(self.masks_dir, self.masks_names[idx]).replace("\\","/")
        image = itk.imread(img_path)
        #seg = itk.imread(seg_path)
        
        if(resampling==True):
            RESimage = resample(image)
            #RESseg = resample(seg)
        
            return RESimage#, RESseg
        else:
            return image#, seg
    
    # P1 .5,.5,.5 sims 10,10,10 -> resample to 1,1,1 dims 5,5,5 -> crop to 0:4, 0:4, 0:4
    # P2 .2,.2,.2 sim 24,24,24  -> resample to 1,1,1, dims 6,6,6 -> crop to 0:4, 0:4, 0:4
    
 