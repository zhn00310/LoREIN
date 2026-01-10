import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class WNNMLoss_wsigma(nn.Module):
    '''
    can use additional index, minimize weighted singular,the weight is related to
    the noise variance
    '''
    def __init__(self, DEVICE, patchSize=5, searchRadius=3, N=3, c=2.8):
        super(WNNMLoss_wsigma, self).__init__()
        self.DEVICE = DEVICE
        self.patchSize = patchSize
        self.searchRadius = searchRadius
        self.N = N
        self.c = c
        
    def gen_patch_batch(self,Img):
        batch, ny, nz = Img.shape[0], Img.shape[1], Img.shape[2]
        Img = Img.unsqueeze(1)
        Patches = F.unfold(Img, (self.patchSize,self.patchSize)).reshape(batch,self.patchSize*self.patchSize,(ny-self.patchSize+1),(nz-self.patchSize+1))
       
        return Patches    
    
    def wnnm_batch(self,noisyImg,index):   
        mm, nn = noisyImg.shape[1],noisyImg.shape[2]
       #noisyImg = self.normalize01(noisyImg,cutValue)
        noisyImg = F.pad(noisyImg,(0,self.patchSize-nn%self.patchSize,0,self.patchSize-mm%self.patchSize,0,0),"constant",0)
        batch, m, n = noisyImg.shape[0],noisyImg.shape[1],noisyImg.shape[2]
        padImg = F.pad(noisyImg,(self.searchRadius,self.searchRadius,self.searchRadius,self.searchRadius, 0, 0),"constant",0)
        Patches = self.gen_patch_batch(padImg)
        Tpatches = Patches.reshape(Patches.shape[0]*Patches.shape[1],1,Patches.shape[2],Patches.shape[3])
        Windows = F.unfold(Tpatches,(2*self.searchRadius+1,2*self.searchRadius+1),stride=self.patchSize)
        Windows = Windows.reshape(batch,self.patchSize*self.patchSize,Windows.shape[1],Windows.shape[2])
               
        currentPatch = F.unfold(noisyImg.unsqueeze(1),(self.patchSize,self.patchSize),stride=self.patchSize).unsqueeze(2)
        
        if index.numel() == 0:
            dist = torch.sum((Windows-currentPatch)**2,1)                
            index = torch.tile(torch.argsort(dist,dim=1).unsqueeze(1),(1,self.patchSize*self.patchSize,1,1))


        Yj = torch.gather(Windows,2,index)[:,:,:self.N,:].permute(0,3,1,2)
        Yj = Yj.reshape(Yj.shape[0]*Yj.shape[1],Yj.shape[2],Yj.shape[3])
        Sigma_n = torch.std(Yj[:,:,0]-torch.mean(Yj,-1),-1,keepdim=True) 
        Sy = torch.linalg.svdvals(Yj)
        Sx = torch.sqrt(torch.maximum(Sy**2-self.N*Sigma_n**2,torch.tensor(0)))
        w = (torch.tensor(self.c)*torch.sqrt(torch.tensor(self.N))/(Sx+torch.tensor(10**(-6))))
        w = w/w[:,-1].unsqueeze(-1).detach()
        sing_val = torch.sum(w.detach()*Sy)
                    
        return sing_val,index