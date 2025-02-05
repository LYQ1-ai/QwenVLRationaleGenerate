import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.nccl.version())
print(torch.distributed.is_nccl_available())