- https://blog.csdn.net/amusi1994/article/details/114559459
- [CoordAttention](https://github.com/Andrew-Qibin/CoordAttention)



---

# **SENet**

![](https://img-blog.csdnimg.cn/img_convert/ee2e7503893ab4d1b801f15035e2b8c0.png)

# CBAM

![](https://img-blog.csdnimg.cn/img_convert/c6fa5c9d7b00eef406e525365303ed2f.png)

# self-attention

![](https://img-blog.csdnimg.cn/img_convert/51f956e9db45aa22a4711ecaa3821eb8.png)

# Coordinate Attention

![](https://img-blog.csdnimg.cn/img_convert/0eaa3720d36e592248ea3f57ce23d4aa.png)

## **CA Block的PyTorch实现**

```python
import torch
from torch import nn
 
 
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()
 
        self.h = h
        self.w = w
 
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
 
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
 
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
 
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
 
        return out
 
 
if __name__ == '__main__':
    x = torch.randn(1, 16, 128, 64)    # b, c, h, w
    ca_model = CA_Block(channel=16, h=128, w=64)
    y = ca_model(x)
    print(y.shape)
```

