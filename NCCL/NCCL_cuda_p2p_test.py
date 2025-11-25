# NCCL_cuda_p2p_test.py
import torch
print("cuda available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
try:
    a = torch.randn(1024,1024).cuda(0)
    b = torch.randn(1024,1024).cuda(1)
    c = a.cuda(1) + b
    print("P2P copy OK")
except Exception as e:
    print("ERROR:", e)
