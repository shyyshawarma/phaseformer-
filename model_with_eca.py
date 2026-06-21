from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from pdb import set_trace as stx

# from utils.antialias import Downsample as downsamp
import cv2


def inv_mag(x):
  fft_ = torch.fft.fft2(x)
  fft_ = torch.fft.ifft2(1*torch.exp(1j*(fft_.angle())))
  return fft_.real

class ECA(nn.Module):
    """Constructs a ECA module.


    Args:
        channels: Number of channels in the input tensor
        b: Hyper-parameter for adaptive kernel size formulation. Default: 1
        gamma: Hyper-parameter for adaptive kernel size formulation. Default: 2 
    """
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()


    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out


    def forward(self, x):

        x1=inv_mag(x)
        # feature descriptor on the global spatial information
        y = self.avg_pool(x1)


        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


        # Multi-scale information fusion
        y = self.sigmoid(y)


        return x * y.expand_as(x)



class MDTA(nn.Module):
	def __init__(self, channels, num_heads):
		super(MDTA, self).__init__()
		self.num_heads = num_heads
		self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

		self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
		self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
		self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

	def forward(self, x):
		b, c, h, w = x.shape
		q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
		  
		q = inv_mag(q)
		k = inv_mag(k)

		q = q.reshape(b, self.num_heads, -1, h * w)
		k = k.reshape(b, self.num_heads, -1, h * w)
		v = v.reshape(b, self.num_heads, -1, h * w)

		q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

		attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
		out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
		return out


class GDFN(nn.Module):
	def __init__(self, channels, expansion_factor):
		super(GDFN, self).__init__()

		hidden_channels = int(channels * expansion_factor)

		self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)

		self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
							  groups=hidden_channels * 2, bias=False)
		
		self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

	def forward(self, x):
		x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
		x = self.project_out(F.gelu(x1) * x2)
		return x

class attention_to_1x1(nn.Module):
	def __init__(self, channels):
		super(attention_to_1x1, self).__init__()
		self.conv1 = nn.Conv2d(channels, channels*2, kernel_size=1, bias=False)
		self.conv2 = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)


	def forward(self,x):
		x=torch.mean(x,-1)
		x=torch.mean(x ,-1)
		x=torch.unsqueeze(x ,-1)
		x=torch.unsqueeze(x ,-1)
		xx = self.conv2(self.conv1(x))    
		b, ch, r, c = x.shape
		# print(ch)
		# exit(0)

		return xx
	

class TransformerBlock(nn.Module):
	def __init__(self, channels, num_heads, expansion_factor):
		super(TransformerBlock, self).__init__()

		self.norm1 = nn.LayerNorm(channels)
		self.attn = MDTA(channels, num_heads)
		self.norm2 = nn.LayerNorm(channels)
		self.ffn = GDFN(channels, expansion_factor)

	def forward(self, x):
		b, c, h, w = x.shape
		x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
						  .contiguous().reshape(b, c, h, w))
		x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
						 .contiguous().reshape(b, c, h, w))
		return x


class DownSample(nn.Module):
	def __init__(self, channels):
		super(DownSample, self).__init__()
		self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
								  nn.PixelUnshuffle(2))

	def forward(self, x):
		return self.body(x)


class UpSample(nn.Module):
	def __init__(self, channels):
		super(UpSample, self).__init__()
		self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
								  nn.PixelShuffle(2))

	def forward(self, x):
		return self.body(x)
  
class UpSample1(nn.Module):
	def __init__(self, channels):
		super(UpSample1, self).__init__()
		self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
								  nn.PixelShuffle(2))

	def forward(self, x):
		return self.body(x)  	


class Restormer(nn.Module):
	def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[16, 32, 64, 128], num_refinement=4,
				 expansion_factor=2.66, ch=[16,16,32,64]):
		
		super(Restormer, self).__init__()
		# self.sig=nn.Sigmoid()

		self.attention = nn.ModuleList([ECA(num_ch) for num_ch in ch])
	   
		self.embed_conv_rgb = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
		self.ups1 = UpSample1(32)

		self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
									   zip(num_blocks, num_heads, channels)])
		# the number of down sample or up sample == the number of encoder - 1
		self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
		self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
		# the number of reduce block == the number of decoder - 1
		self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
									  for i in reversed(range(2, len(channels)))])
		# the number of decoder == the number of encoder - 1
		self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
													   for _ in range(num_blocks[2])])])
		self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
											 for _ in range(num_blocks[1])]))
		# the channel of last one is not change
		self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]))

		self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
										  for _ in range(num_refinement)])
		self.output = nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=False)
		self.output1= nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)
								 
		self.ups2 = UpSample1(16)
		self.outputl=nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False)
								 
	def forward(self,RGB_input):

		fo_rgb = self.embed_conv_rgb(RGB_input)

		out_enc_rgb1 = self.encoders[0](fo_rgb)
		out_enc_rgb2 = self.encoders[1](self.downs[0](out_enc_rgb1))
		out_enc_rgb3 = self.encoders[2](self.downs[1](out_enc_rgb2))
		out_enc_rgb4 = self.encoders[3](self.downs[2](out_enc_rgb3))
		
	  

		out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc_rgb4), self.attention[0](out_enc_rgb3)], dim=1)))

		out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3),self.attention[1](out_enc_rgb2)], dim=1)))

		fd = self.decoders[2](torch.cat([self.ups[2](out_dec2),self.attention[2](out_enc_rgb1)], dim=1))

		fr = self.refinement(fd)  
		#fr is final output 
		
		outi=self.ups1(fr) 

		return self.output(self.outputl(fr)),self.output(self.output1(outi))