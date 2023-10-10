import torch
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
for n, m in m.named_modules():
    print(n)

for n, m in model.named_modules():
    print(n)