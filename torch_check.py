# import torch
# print("torch:", torch.__version__)
# print("cuda runtime:", torch.version.cuda)
# print("device name:", torch.cuda.get_device_name(0))
# print("capability:", torch.cuda.get_device_capability(0))

import timm
# timms version
print("version:", timm.__version__)
print("MobileNets:", timm.list_models('*mobilenetv4*'))
# print("EfficientNets:", timm.list_models('*efficientnet*'))