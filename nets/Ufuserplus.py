import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch import einsum
class Transformer_block(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Transformer_block, self).__init__()
        self.embed=nn.Conv2d(in_dim, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads = 8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN=nn.Conv2d(out_dim*2, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")          
    def forward(self, x):
        x=self.embed(x)
        x1=self.GlobalFeature(x)
        x2=self.LocalFeature(x)
        out=self.FFN(torch.cat((x1,x2),1))
        return out
class GlobalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(GlobalFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LocalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=2,
                 ):
        super(LocalFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(*[ResBlock(dim,dim) for i in range(num_blocks)])
        # self.Extraction = nn.Sequential(*[HjmBlock(dim) for i in range(num_blocks)])
    def forward(self, x):
        return self.Extraction(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
        )
    def forward(self, x):
        out = self.conv(x)
        return out+x

class LFblock(nn.Module):
    def __init__(self, n_feats):
        super(LFblock, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.conv2=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False))
        self.gates = nn.Parameter(torch.zeros(2))  # 初始均分  
    def forward(self,x):
        # 这样的门控感觉更加真
        w = torch.softmax(self.gates, dim=0)       # w[0]+w[1]=1
        y = w[0] * self.conv1(x) + w[1] * self.conv2(x)
        return y

class HjmBlock(nn.Module):
    def __init__(self, in_channels):
        super(HjmBlock, self).__init__()
        self.HFconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
        )
        self.encoder = ResBlock(in_channels,in_channels)
        self.LFdecoder = LFblock(in_channels)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.ega=selfAttention(in_channels, in_channels)
        self.raw_alpha=nn.Parameter(torch.ones(1))
        self.raw_alpha.data.fill_(0)
        self.linear1 = nn.Conv2d(in_channels*2,in_channels,3,1,1,bias=False)
        self.linear2 = nn.Conv2d(in_channels,in_channels,1,1,0,bias=False)
        self.CAatt = CALayer(in_channels)
        self.norm = LayerNorm(in_channels,'BiasFree')
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=False)
        high = high + torch.tanh(self.raw_alpha) * self.ega(high, high)
        highFeature = self.HFconv(high)
        lowFeature = self.LFdecoder(x2)
        x3 = F.interpolate(lowFeature, size=x.size()[-2:], mode='bilinear', align_corners=False)
        out = self.linear2(self.CAatt(self.linear1(torch.cat([x3, highFeature], dim=1))))
        out = self.norm(out)
        return out + x

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):

        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 out_fratures,
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias,padding_mode="reflect")

        self.project_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
##########################################################################    
class invertedBlock(nn.Module):
    def __init__(self, in_channel, out_channel,ratio=2):
        super(invertedBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.GELU()
        ## 7*7卷积，并行3*3卷积
        self.conv1 = nn.Conv2d(internal_channel, internal_channel, 7, 1, 3, groups=in_channel,bias=False)

        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=out_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.pw1 = nn.Conv2d(in_channels=in_channel, out_channels=internal_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channel, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
    def hifi(self,x):
        x1=self.pw1(x)
        x1=self.relu(x1)
        x1=self.conv1(x1)
        x1=self.relu(x1)
        x1=self.pw2(x1)
        x1=self.relu(x1)
        # x2 = self.conv2(x)
        x3 = x1+x
        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.layer_norm(x3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = self.convFFN(x3)
        return x4
    def forward(self, x):
        return self.hifi(x)+x
    # 卷积版前馈网络
class ConvFFN(nn.Module):

    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()

        internal_channels = in_channels * expend_ratio
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        x1 = self.pw1(x)
        x2 = self.nonlinear(x1)
        x3 = self.pw2(x2)
        x4 = self.nonlinear(x3)
        return x4 + x
# 这个就是MBB
class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.conv2=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.alpha*self.conv1(x)+self.beta*self.conv2(x)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, max(channel // reduction, 1), 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channel // reduction, 1), channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # 这东西作为权重值之后乘以原本的x就广播城H*W了
        return x * y
class Downupblock(nn.Module):
    def __init__(self, n_feats):
        super(Downupblock, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)  # nn.Sequential(one_module(n_feats),

        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))
        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)  # one_module(n_feats)
        # 池化步骤
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))

        self.raw_alpha.data.fill_(0)
        self.ega=selfAttention(n_feats, n_feats)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)

        high=high+self.ega(high,high)*self.raw_alpha
        x2=self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x
# 这一整个应该都是高频增强模块的内容
class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        # encoder和decoder都是一个MBB
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)  # nn.Sequential(one_module(n_feats),
        #                     one_module(n_feats),
        #                     one_module(n_feats))
        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))

        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)  # one_module(n_feats)
        # 3×3 卷积 + 步长1 + padding1，就是最常见的 “保持尺寸不变的卷积
        # 还真是，1*1卷积每个像素点会从所有通道上获取信息，而3*3会从通道和周边获取信息
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))
        # fill 0
        self.raw_alpha.data.fill_(0)
        self.ega=selfAttention(n_feats, n_feats)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        # x2是经过池化之后的低频特征
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        # high是经过减法之后的高频特征
        # 之后再过一个自注意力
        high=high+self.ega(high,high)*self.raw_alpha
        # 三个MBB，从这开始输入频率融合模块
        x2=self.decoder_low(x2)
        x3 = x2
        # 从这开始高频输入频率模块
        high1 = self.decoder_high(high)
        # x4是低频上采样之后的结果
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        # 最后拼接再过卷积和CA，CA是池化卷积激活
        # 通道注意力之后再接一个残差
        
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x
class basic_block(nn.Module):
    ## 双并行分支，通道分支和空间分支
    def __init__(self, in_channel, out_channel, depth = 1,ratio=1):
        super(basic_block, self).__init__()
        # 个数为depth个
        self.rep1 = nn.Sequential(*[invertedBlock(in_channel=in_channel, out_channel=in_channel,ratio=ratio) for i in range(depth)])
        self.relu=nn.GELU()
        # 一部分做3个3*3卷积，一部分做1个
        self.updown=Updownblock(in_channel)
        self.downup=Downupblock(in_channel)
        self.FFN=nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3,stride=1,padding=1,bias=False,padding_mode="reflect")
    def forward(self, x):
        # x1 = self.rep1(x)
        x1=self.updown(x)
        x1=self.downup(x1)
        # x2 = self.FFN(x1+x)
        return x1+x
class selfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(selfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, feature, feature_map):
        query = self.query_conv(feature)
        key = self.key_conv(feature)
        value = self.value_conv(feature)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale

        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)

        output_feature_map = (feature_map + attended_values)

        return output_feature_map

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# class Ufuserplus(nn.Module):
#     def __init__(self, 
#                 channel_heads = [1,2,4,4],
#                 spatial_heads = [2,2,3,3],            
#                 overlap_ratio=[0.5, 0.5, 0.5,0.5],
#                 window_size = 4,
#                 spatial_dim_head = 16,
#                 bias = False,
#                 ffn_expansion_factor = 1,
#                 LayerNorm_type = 'BiasFree',
                 
#                  ):
#         super(Ufuserplus, self).__init__()
        
#         channel=[8,16,32,32]
#         depth=[2,3,4,2]

#         self.V_en_1 = Transformer_block(1, channel[0])
#         self.V_en_2 = Transformer_block(channel[0], channel[1])
#         self.V_en_3 = Transformer_block(channel[1], channel[2])
#         self.V_en_4 = Transformer_block(channel[2], channel[3])

#         self.V_ff_1 = basic_block(channel[0], channel[0],depth[0])
#         self.V_ff_2 = basic_block(channel[1], channel[1],depth[1])
#         self.V_ff_3 = basic_block(channel[2], channel[2],depth[2])
#         self.V_ff_4 = basic_block(channel[3], channel[3],depth[3])

#         self.fl_1 = basic_block(channel[0], channel[0],depth[0])
#         self.fl_2 = basic_block(channel[1], channel[1],depth[1])
#         self.fl_3 = basic_block(channel[2], channel[2],depth[2])
#         self.fl_4 = basic_block(channel[3], channel[3],depth[3])


#         self.I_ff_1 = basic_block(channel[0], channel[0],depth[0])
#         self.I_ff_2 = basic_block(channel[1], channel[1],depth[1])
#         self.I_ff_3 = basic_block(channel[2], channel[2],depth[2])
#         self.I_ff_4 = basic_block(channel[3], channel[3],depth[3])

#         self.I_en_1 = Transformer_block(1, channel[0])
#         self.I_en_2 = Transformer_block(channel[0], channel[1])
#         self.I_en_3 = Transformer_block(channel[1], channel[2])
#         self.I_en_4 = Transformer_block(channel[2], channel[3])

#         self.f_1 = Transformer_block(channel[0]*2, channel[0])
#         self.f_2 = Transformer_block(channel[1]*2, channel[1])
#         self.f_3 = Transformer_block(channel[2]*2, channel[2])
#         self.f_4 = Transformer_block(channel[3]*2, channel[3])

#         self.V_down1=nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
#         self.V_down2=nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
#         self.V_down3=nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        

#         self.I_down1=nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
#         self.I_down2=nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
#         self.I_down3=nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        

#         self.up4=nn.Sequential(
#             nn.ConvTranspose2d(channel[3],channel[2], 4, 2, 1, bias=False),
#             nn.ReLU()
#         )
#         self.up3=nn.Sequential(
#             nn.ConvTranspose2d(channel[2],channel[1], 4, 2, 1, bias=False),
#             nn.ReLU()
#         )
#         self.up2=nn.Sequential(
#             nn.ConvTranspose2d(channel[1],channel[0], 4, 2, 1, bias=False),
#             nn.ReLU()
#         )

#         self.de_1 = Transformer_block(channel[0]*2,channel[0])
#         self.de_2 = Transformer_block(channel[1]*2,channel[1])
#         self.de_3 = Transformer_block(channel[2]*2,channel[2])
#         self.de_4 = Transformer_block(channel[3],channel[3])

#         # self.Encon = Transformer_block(2, channel[0])
#         # self.Decon = Transformer_block(channel[0], channel[0])


#         self.last = nn.Sequential(
#             nn.Conv2d(channel[0], 1, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
#             nn.Sigmoid()
#         )
#         # self.last_out = nn.Sequential(
#         #     nn.Conv2d(channel[0], 1, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
#         #     nn.Sigmoid()
#         # )
#     def forward(self, i,v):
#         # I_en_1是Restormer_CNN_block模块
#         #  down为一层二维卷积
#         i_1=self.I_en_1(i)
#         i_2=self.I_en_2(self.I_down1(i_1))
#         i_3=self.I_en_3(self.I_down2(i_2))
#         i_4=self.I_en_4(self.I_down3(i_3))

#         i_f1 = self.I_ff_1(i_1)
#         i_f2 = self.I_ff_2(i_2)
#         i_f3 = self.I_ff_3(i_3)
#         i_f4 = self.I_ff_4(i_4)

#         v_1=self.V_en_1(v)
#         v_2=self.V_en_2(self.V_down1(v_1))
#         v_3=self.V_en_3(self.V_down2(v_2))
#         v_4=self.V_en_4(self.V_down3(v_3))

#         v_f1 = self.V_ff_1(v_1)
#         v_f2 = self.V_ff_2(v_2)
#         v_f3 = self.V_ff_3(v_3)
#         v_f4 = self.V_ff_4(v_4)

#         f_1=self.f_1(torch.cat((i_f1,v_f1),1))
#         f_2=self.f_2(torch.cat((i_f2,v_f2),1))
#         f_3=self.f_3(torch.cat((i_f3,v_f3),1))
#         f_4=self.f_4(torch.cat((i_f4,v_f4),1))

#         f_1 = self.fl_1(f_1)
#         f_2 = self.fl_2(f_2)
#         f_3 = self.fl_3(f_3)
#         f_4 = self.fl_4(f_4)

#         out=self.up4(self.de_4(f_4))
#         out=self.up3(self.de_3(torch.cat((out,f_3),1)))
#         out=self.up2(self.de_2(torch.cat((out,f_2),1)))
#         out=self.de_1(torch.cat((out,f_1),1))

#         return self.last(out)

class Ufuserplus(nn.Module):
    def __init__(self, 
                channel_heads = [1,2,4,4],
                spatial_heads = [2,2,3,3],            
                overlap_ratio=[0.5, 0.5, 0.5,0.5],
                window_size = 4,
                spatial_dim_head = 16,
                bias = False,
                ffn_expansion_factor = 1,
                LayerNorm_type = 'BiasFree',
                 
                 ):
        super(Ufuserplus, self).__init__()
        
        channel=[8,16,32,32]


        self.V_en_1 = Transformer_block(1, channel[0])
        self.V_en_2 = Transformer_block(channel[0], channel[1])
        self.V_en_3 = Transformer_block(channel[1], channel[2])
        self.V_en_4 = Transformer_block(channel[2], channel[3])

        self.I_en_1 = Transformer_block(1, channel[0])
        self.I_en_2 = Transformer_block(channel[0], channel[1])
        self.I_en_3 = Transformer_block(channel[1], channel[2])
        self.I_en_4 = Transformer_block(channel[2], channel[3])

        self.f_1 = Transformer_block(channel[0]*2, channel[0])
        self.f_2 = Transformer_block(channel[1]*2, channel[1])
        self.f_3 = Transformer_block(channel[2]*2, channel[2])
        self.f_4 = Transformer_block(channel[3]*2, channel[3])

        self.V_down1=nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.V_down2=nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.V_down3=nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        

        self.I_down1=nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.I_down2=nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.I_down3=nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        

        self.up4=nn.Sequential(
            nn.ConvTranspose2d(channel[3],channel[2], 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up3=nn.Sequential(
            nn.ConvTranspose2d(channel[2],channel[1], 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up2=nn.Sequential(
            nn.ConvTranspose2d(channel[1],channel[0], 4, 2, 1, bias=False),
            nn.ReLU()
        )

        self.de_1 = Transformer_block(channel[0]*2,channel[0])
        self.de_2 = Transformer_block(channel[1]*2,channel[1])
        self.de_3 = Transformer_block(channel[2]*2,channel[2])
        self.de_4 = Transformer_block(channel[3],channel[3])

        self.last = nn.Sequential(
            nn.Conv2d(channel[0], 1, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
            nn.Sigmoid()
        )
    def forward(self, v,i):
        # I_en_1是Restormer_CNN_block模块
        #  down为一层二维卷积
        i_1=self.I_en_1(i)
        i_2=self.I_en_2(self.I_down1(i_1))
        i_3=self.I_en_3(self.I_down2(i_2))
        i_4=self.I_en_4(self.I_down3(i_3))

        v_1=self.V_en_1(v)
        v_2=self.V_en_2(self.V_down1(v_1))
        v_3=self.V_en_3(self.V_down2(v_2))
        v_4=self.V_en_4(self.V_down3(v_3))

        f_1=self.f_1(torch.cat((i_1,v_1),1))
        f_2=self.f_2(torch.cat((i_2,v_2),1))
        f_3=self.f_3(torch.cat((i_3,v_3),1))
        f_4=self.f_4(torch.cat((i_4,v_4),1))

        out=self.up4(self.de_4(f_4))
        out=self.up3(self.de_3(torch.cat((out,f_3),1)))
        out=self.up2(self.de_2(torch.cat((out,f_2),1)))
        out=self.de_1(torch.cat((out,f_1),1))

        return self.last(out)
