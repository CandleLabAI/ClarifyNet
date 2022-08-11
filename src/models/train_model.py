# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

from dataloader import *
from tqdm import tqdm
from network import *
from config import *
from utils import *
from loss import *
import os

def main():
    """ training ClarifyNet model 
    """
    train_loader, val_loader, test_loader = getDataLoader(batch_size=BATCH_SIZE)
    model = getModel()
    model = nn.DataParallel(model)
    model.to(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = INITIAL_LEARNING_RATE)
    criterian = Loss().to(DEVICE)
    
    for epoch in range(EPOCHS):
        print('Epoch: {}'.format(epoch))
        l_train = []
        ps_train = []
        l_test = []
        ps_test = []
        mss = []
        model.train()
        global current_psnr

        for i, data in tqdm(enumerate(train_loader)):
            x, y1, y2, y3 = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE), data[3].to(DEVICE)
            optimizer.zero_grad()
            pred1, pred2, pred3 = model(x)
            loss = criterian(pred1, pred2, pred3, y1, y2, y3)
             
            psnr1 = getPSNR(pred3, y3)
            loss.backward()
            optimizer.step()
            l_train.append(loss.item())
            ps_train.append(psnr1)
        print("Epoch loss: ", sum(l_train)/len(l_train))
        print('Epoch {} PSNR: '.format(epoch), sum(ps_train)/len(ps_train))

        with torch.no_grad():
            model.eval()
            mss_val = []
            for i, data in tqdm(enumerate(test_loader)):
                x, y1, y2, y3 = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE), data[3].to(DEVICE)
                pred1, pred2, pred3 = model(x)

                psnr1 = getPSNR(pred3,y3)
                val_mss = ssim(pred3, y3)
                mss_val.append(val_mss)
                l_test.append(loss.item())
                ps_test.append(psnr1)
        print("Val Epoch loss: ", sum(l_test)/len(l_test))
        print('Val Epoch {} PSNR: '.format(epoch), sum(ps_test)/len(ps_test))
        print('VAL SSIM: ', sum(mss_val)/len(mss_val))

        if current_psnr < sum(ps_test) / len(ps_test):
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict()
            } 
            print("saving best one.....")
            torch.save(checkpoint, os.path.join(WEIGHTS_DIR, "best.pth")
            current_psnr = sum(ps_test)/len(ps_test)
          
        if (epoch + 1) % 50 == 0:
            checkpoint = {
            'weights': model.state_dict(),
            'optimizer':optimizer.state_dict()
            } 
            torch.save(checkpoint, os.path.join(WEIGHTS_DIR, "model{}.pth".format(epoch + 1))

if __name__ == '__main__':
    
    main()
