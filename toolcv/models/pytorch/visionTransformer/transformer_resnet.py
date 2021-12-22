import torch
from torch import nn,einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class FeedForward(nn.Module):
    def __init__(self, in_c,hide_c, out_c, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, hide_c, 1),
            nn.BatchNorm2d(hide_c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hide_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c),
        ) if in_c != out_c else nn.Sequential()

        self.relu = nn.GELU()

    def forward(self, x):
        return self.relu(self.net(x)+self.downsample(x))

class Attention2(nn.Module):
    def __init__(self, in_c,hide_c,out_c,stride=1, dropout = 0.1,scale=1/8):
        super().__init__()
        self.scale = scale #dim_key ** -0.5
        # self.scale = out_c ** -0.5
        hide_c2 = hide_c//2
        self.to_q = nn.Sequential(nn.Conv2d(in_c, hide_c2, 1, stride = stride, bias = False), nn.BatchNorm2d(hide_c2))
        self.to_k = nn.Sequential(nn.Conv2d(in_c, hide_c2, 1, bias = False), nn.BatchNorm2d(hide_c2))
        self.to_v = nn.Sequential(nn.Conv2d(in_c, hide_c, 1, bias = False), nn.BatchNorm2d(hide_c))

        self.attend = nn.Softmax(dim = -1)

        self.to_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hide_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1,stride=stride),
            nn.BatchNorm2d(out_c)
        ) if (stride>1 or in_c!=out_c) else nn.Sequential()

        self.relu = nn.GELU() #nn.ReLU(inplace=True)

    def forward(self, x):
        q = self.to_q(x)
        b, c, h, w = q.shape
        # qkv = (q, self.to_k(x), self.to_v(x))
        # q, k, v = map(lambda t: rearrange(q,'b c1 h w -> b c1 (h w)',c1=c1), qkv)
        k = self.to_k(x)
        v = self.to_v(x)
        q = rearrange(q,'b c1 h w -> b c1 (h w)',c1=q.shape[1])
        k = rearrange(k,'b c1 h w -> b c1 (h w)',c1=k.shape[1])
        v = rearrange(v,'b c1 h w -> b c1 (h w)',c1=v.shape[1])

        dots = einsum('b c i, b c j -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b d j -> b i d', attn, v)
        out = rearrange(out, 'b (h w) d -> b d h w', h = h, w = w)
        out = self.to_out(out)

        return self.relu(out+self.downsample(x))

class Attention(nn.Module):
    def __init__(self, in_c,hide_c,out_c,stride=1, dropout = 0.1,scale=1/8):
        super().__init__()
        self.scale = scale #dim_key ** -0.5
        # self.scale = out_c ** -0.5
        self.winsize = 7
        hide_c2 = hide_c//2
        self.to_q = nn.Sequential(nn.Conv2d(in_c, hide_c2, 1, stride = stride, bias = False), nn.BatchNorm2d(hide_c2))
        self.to_k = nn.Sequential(nn.Conv2d(in_c, hide_c2, 1, bias = False), nn.BatchNorm2d(hide_c2))
        self.to_v = nn.Sequential(nn.Conv2d(in_c, hide_c, 1, bias = False), nn.BatchNorm2d(hide_c))

        self.attend = nn.Softmax(dim = -1)

        self.to_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hide_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1,stride=stride),
            nn.BatchNorm2d(out_c)
        ) if (stride>1 or in_c!=out_c) else nn.Sequential()

        self.relu = nn.GELU() #nn.ReLU(inplace=True)

    def forward(self, x):
        q = self.to_q(x)
        b, c, h, w = q.shape
        h1 = h//self.winsize
        w1 = w//self.winsize
        # qkv = (q, self.to_k(x), self.to_v(x))
        # q, k, v = map(lambda t: rearrange(q,'b c1 h w -> b c1 (h w)',c1=c1), qkv)
        k = self.to_k(x)
        v = self.to_v(x)
        q = rearrange(q,'b c (h1 ws1) (w1 ws2) -> b (c ws1 ws2) (h1 w1)',ws1=self.winsize,ws2=self.winsize)
        k = rearrange(k,'b c (h1 ws1) (w1 ws2) -> b (c ws1 ws2) (h1 w1)',ws1=self.winsize,ws2=self.winsize)
        v = rearrange(v,'b c1 (h1 ws1) (w1 ws2) -> b (c1 ws1 ws2) (h1 w1)',c1=v.shape[1],ws1=self.winsize,ws2=self.winsize)

        dots = einsum('b c i, b c j -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b d j -> b i d', attn, v)
        # out = rearrange(out, 'b (h w) d -> b d h w', h = h, w = w)
        out = rearrange(out, 'b (h1 w1) (c1 ws1 ws2) -> b c1 (h1 ws1) (w1 ws2)',
                        ws1=self.winsize,ws2=self.winsize,h1=h1,w1=w1)
        out = self.to_out(out)

        return self.relu(out+self.downsample(x))

class TransformerBlock(nn.Module):
    def __init__(self,in_c,hide_c,out_c,ksize=3,stride=1,use_se=False,dropout=0.1):
        super().__init__()
        self.attention = Attention(in_c,hide_c,out_c,stride,dropout)
        self.feedforward = FeedForward(out_c,out_c//4,out_c,dropout)

    def forward(self,x):
        x = self.attention(x)
        x = self.feedforward(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.flatten(x,1)

def _make_layer(self, block, in_c,hide_c,out_c, blocks, stride=1,dropout=0.1):
    layers = []
    for i in range(blocks):
        if i == 0:
            layers.append(block(in_c,hide_c,out_c,3,stride,self.use_se,dropout))
        else:
            layers.append(block(out_c, hide_c, out_c, 3, 1,self.use_se,dropout))

    return nn.Sequential(*layers)

def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class ResNet(nn.Module):
    """
    layers = [2,2,2,2] 18
    layers = [3,4,6,3] 34
    layers = [3,4,6,3] 50
    layers = [3,4,23,3] 101
    layers = [3,8,36,3] 152
    """
    def __init__(self, block,param={}, num_classes=1000,dropout=0.2,use_se=False):
        super().__init__()
        layers = param["layers"]
        in_c = param["in_c"]
        hide_c = param["hide_c"]
        out_c = param["out_c"]
        self.use_se = use_se

        self.stem = nn.Sequential(
            nn.Conv2d(3,64,3,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )

        self.layer1 = _make_layer(self,block, in_c[0],hide_c[0],out_c[0], layers[0],1,dropout)
        self.layer2 = _make_layer(self,block, in_c[1],hide_c[1],out_c[1], layers[1], 2,dropout)
        self.layer3 = _make_layer(self,block, in_c[2],hide_c[2],out_c[2], layers[2], 2,dropout)
        self.layer4 = _make_layer(self,block, in_c[3],hide_c[3],out_c[3], layers[3], 2,dropout)

        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_c[-1], num_classes)
        )

        _init_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if isinstance(x,tuple):x=x[0]
        x = self.classify(x)

        return x


def resnet50(block,num_classes=1000,dropout=0.2,use_se=False):
    """
    26: (Bottleneck, (1, 2, 4, 1)),
    38: (Bottleneck, (2, 3, 5, 2)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3))

    """

    # param = {
    #     "layers": [3, 4, 6, 3],
    #     "in_c": [64, 256, 512, 1024],
    #     "hide_c": [64, 128, 256, 512],
    #     "out_c": [256, 512, 1024, 2048],
    # }

    param = {
        "layers": [3, 4, 6, 3],
        "in_c": [64, 128, 256, 512],
        "hide_c": [64, 128, 256, 512],
        "out_c": [128, 256, 512, 1024],
    }

    return ResNet(block, param, num_classes, dropout, use_se)

if __name__ == "__main__":
    x = torch.rand([1,3,224,224])
    m = resnet50(TransformerBlock)
    print(m)
    pred = m(x)
    print(pred.shape)

    torch.save(m.state_dict(),"resnet50.pth")