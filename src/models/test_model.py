# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

from dataloader import *
sys.path.append("../src")
from network import *
from config import *
from utils import *
from loss import *

def main():
    """ testing performance of model. 
    """
    model = getModel()
    model = nn.DataParallel(model)
    model.to(DEVICE)
    net = torch.load(WEIGHTS_DIR)
    model.load_state_dict(net['state_dict'])

    _, _ ,test_loader = getDataLoader(batch_size=1)
    psnr = []
    Ssim = []


    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data[0].to(DEVICE)
            y = data[-1].to(DEVICE)
            yhat = model(x)

            p = getPSNR(yhat, y)
            s = ssim(yhat, y)

            psnr.append(p.item())
            Ssim.append(s.item())

    print("PSNR: ", sum(psnr)/len(psnr))
    print("SSIM: ", sum(Ssim)/len(Ssim))

if __name__ == '__main__':
    main()
