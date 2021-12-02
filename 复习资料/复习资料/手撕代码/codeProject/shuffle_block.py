import torch
from torch import nn

class shuffle_block(nn.Module):
    def __init__(self):
        super(shuffle_block, self).__init__()
        self.conv1 = nn.Conv2d(3,9,kernel_size=1,padding=0)
        self.dw_conv = nn.Conv2d(9,9,kernel_size=3,padding=1,groups=9)
        self.conv2 = nn.Conv2d(9,3,kernel_size=1,padding=0)

    def forward(self,x,group):
        x = self.conv1(x)
        n,c,h,w = x.shape
        x = x.reshape(n,group,c//group,h,w)
        x = x.permute(0,2,1,3,4)
        x = x.reshape(n,c,h,w)
        x = self.dw_conv(x)
        x = self.conv2(x)

        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 50, 50)
    net = shuffle_block()
    y = net(x,3)

    print(y.shape)