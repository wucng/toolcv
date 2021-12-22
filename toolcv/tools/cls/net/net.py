from torch import nn
# from timm.models import *
# from torchvision.models import *

def _initParmas(modules, std=0.01, mode='normal'):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if mode == 'normal':
                nn.init.normal_(m.weight, std=std)
            elif mode == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Linear):
        #     nn.init.normal_(m.weight, 0, std=std)
        #     if m.bias is not None:
        #         # nn.init.zeros_(m.bias)
        #         nn.init.constant_(m.bias, 0)
