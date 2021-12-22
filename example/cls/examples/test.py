from timm.models.resnest import resnest50d
from torch import nn

model = resnest50d(True)
for parma in model.parameters():
    parma.requires_grad_(False)
for parma in model.layer4.parameters():
    parma.requires_grad_(True)
model.fc = nn.Linear(model.num_features,num_classes)