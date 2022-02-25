import torch
import torch.nn as nn


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, y):
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        x = (1-sim_map)*x + sim_map*y
        
        return x

# before relu        
def PagFF(x, y, with_channel=False):
    
    if with_channel:
        sim_map = torch.sigmoid(x * y)
    else:
        sim_map = torch.sigmoid(torch.sum(x * y, dim=1))
        
    x = (1-sim_map)*x + sim_map*y
    
    return x


# must before relu
class BagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, with_relu=False, BatchNorm=nn.BatchNorm2d):
        super(BagFM, self).__init__()
        self.with_relu = with_relu
        self.conv_p = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_d = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.f_p = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_i = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_d = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_relu:
            self.relu = nn.ReLU(inplace=True)
            self.bn = BatchNorm(in_channels)
        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        
        if self.with_relu:
            p = self.relu(p)
            i = self.relu(i)
            d = self.relu(self.bn(d))
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(edge_att*p + i)
        
        d_q = self.f_d(d)
        i_k = self.f_i(i)
        p_k = self.f_p(p)
        sim_map = torch.softmax(torch.cat([torch.sum(d_q*i_k, dim=1).unsqueeze(1), torch.sum(d_q*p_k, dim=1).unsqueeze(1)], dim=1), dim=1)
        d_add = sim_map[:,:-1,:,:]*i + sim_map[:,1:,:,:]*p + d
        d_add = self.conv_d(d)
        
        return p_add + i_add + d_add

class DFM(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DFM, self).__init__()

        self.conv = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=3, padding=1, bias=False)                  
                                )

        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att*p + (1-edge_att)*i)
    
    

if __name__ == '__main__':

    
    x = torch.rand(4, 64, 32, 64).cuda()
    y = torch.rand(4, 64, 32, 64).cuda()
    z = torch.rand(4, 64, 32, 64).cuda()
    net = PagFM(64, 16, with_channel=True).cuda()
    
    out = net(x,y)