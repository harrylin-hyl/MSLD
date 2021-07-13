import math
import random
import numpy as np
import torch
import torch.nn as nn

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, c_ratio=8, freq_sel_method = 'top16', mode='MS'):
        super(MultiSpectralAttentionLayer, self).__init__()
        assert mode == 'MS' or mode == 'SE'
        self.mode = mode
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.c_ratio = c_ratio

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(1)
        self.conv = nn.Conv2d(channel//c_ratio, channel//c_ratio, kernel_size=1)

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if self.mode == 'SE':
            y = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(n,c)
        else:
            if h != self.dct_h or w != self.dct_w:
                x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
                # If you have concerns about one-line-change, don't worry.   :)
                # In the ImageNet models, this line will never be triggered. 
                # This is for compatibility in instance segmentation and object detection.
            y = self.dct_layer(x_pooled)
        # block
        y = self.fc(y).view(n, self.c_ratio, c//self.c_ratio, 1, 1)
        x = x.view(n, self.c_ratio, c//self.c_ratio, h, w)
        compress_x = (x * y.expand_as(x)).sum(1)
        compress_x = self.conv(compress_x)
        return compress_x

        # # random
        # random.seed(255)
        # random_ind = torch.arange(0, c, 1)
        # random.shuffle(random_ind)
        # x_random = x[:,random_ind.long(),::]
        # y_random = y[:,random_ind.long()]
        # y = self.fc(y_random).view(n, self.c_ratio, c//self.c_ratio, 1, 1)
        # x = x_random.view(n, self.c_ratio, c//self.c_ratio, h, w)
        # compress_x = (x * y.expand_as(x)).sum(1)
        # compress_x = self.conv(compress_x)
        # return compress_x

        # sort 
        # _, sort_ind = torch.sort(y, dim=1, descending=True)
        # group_ind = sort_ind.view(n, self.c_ratio, c//self.c_ratio)
        # inverse_ind = torch.arange((c//self.c_ratio)-1, -1, -1)
        # group_ind[:,1, :] = group_ind[:,1, inverse_ind]
        # group_ind[:,3, :] = group_ind[:,3, inverse_ind]
        # group_ind[:,5, :] = group_ind[:,5, inverse_ind]
        # group_ind[:,7, :] = group_ind[:,7, inverse_ind]
        # group_ind = group_ind.view(n, -1)
        # x_sort = torch.zeros_like(x)
        # for i in range(n):
        #     x_sort[i] = x[i, group_ind[i,:],:,:]
        # y_sort = y.gather(dim=1, index=group_ind)
        # y_group = y_sort.view(n, self.c_ratio, c//self.c_ratio, 1, 1)
        # x_group = x_sort.view(n, self.c_ratio, c//self.c_ratio, h, w)

        # y_group_weight = self.softmax(y_group)
        # compress_x = (x_group * y_group_weight).sum(1)
        # compress_x = self.conv(compress_x)

        # return compress_x


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, width, height, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

class SingleMultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_n, reduction = 4, n_ratio=4, mode='MS'):
        super(SingleMultiSpectralAttentionLayer, self).__init__()
        assert mode == 'MS' or mode == 'SE'
        self.mode = mode
        self.reduction = reduction
        self.dct_n = dct_n
        self.c_ratio = n_ratio

        mapper_n = np.arange(0, dct_n)
        self.num_split = len(mapper_n)
        self.dct_layer = SingleMultiSpectralDCTLayer(dct_n, mapper_n, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.conv = nn.Conv1d(channel//n_ratio, channel//n_ratio, kernel_size=1)

    def forward(self, x):
        b,c,n = x.shape
        x_pooled = x
        if self.mode == 'SE':
            y = torch.nn.functional.adaptive_avg_pool1d(x, 1).view(b,c)
        else:
            if n != self.dct_n:
                x_pooled = torch.nn.functional.adaptive_avg_pool1d(x, (self.dct_n))
                # If you have concerns about one-line-change, don't worry.   :)
                # In the ImageNet models, this line will never be triggered. 
                # This is for compatibility in instance segmentation and object detection.
            y = self.dct_layer(x_pooled)

        # y = self.fc(y).view(b, self.c_ratio, c//self.c_ratio, 1)
        # x = x.view(b, self.c_ratio, c//self.c_ratio, n)
        # compress_x = (x * y.expand_as(x)).sum(1)
        # compress_x = self.conv(compress_x)

        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)

        return x
        # # sort 
        # _, sort_ind = torch.sort(y, dim=1, descending=True)
        # sort_ind = sort_ind.cpu().numpy()
        # group_ind = np.resize(sort_ind, (n, c_ratio, c//c_ratio, 1, 1))
        # group_ind[:,1, :,:,:] = group_ind[:,1, ::-1,:,:]
        # group_ind[:,3, :,:,:] = group_ind[:,3, ::-1,:,:]
        # group_ind[:,5, :,:,:] = group_ind[:,5, ::-1,:,:]
        # group_ind[:,7, :,:,:] = group_ind[:,7, ::-1,:,:]
        # group_ind = torch.from_numpy(np.resize(group_ind, (n, c, 1, 1))).cuda()

        # y_sort = y.gather(dim=1, index=group_ind)
        # y_group = y_sort.view(n, c_ratio, c//c_ratio, 1, 1)

        # x_sort = torch.zeros_like(x)
        # for i in range(n):
        #     x_sort[i] = x[i, group_ind[i,:,0,0],:,:]
        
        # x_group = x_sort.view(n, c_ratio, c//c_ratio, h, w)

        # y_group_weight = self.softmax(y_group)

        # return (x_group * y_group_weight).sum(1)


class SingleMultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, sample, mapper_n, channel):
        super(SingleMultiSpectralDCTLayer, self).__init__()
        
        assert channel % len(mapper_n) == 0

        self.num_freq = len(mapper_n)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(sample, mapper_n, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 3, 'x must been 3 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=2)
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_n, mapper_n, channel):
        dct_filter = torch.zeros(channel, tile_size_n)

        c_part = channel // len(mapper_n)

        for i, u_n in enumerate(mapper_n):
            for t_n in range(tile_size_n):
                dct_filter[i * c_part: (i+1)*c_part, t_n] = self.build_filter(t_n, u_n, tile_size_n)
                        
        return dct_filter