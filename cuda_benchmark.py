import torch
import time

# Matrix multiply on CPU
a = torch.randn(10000, 10000)
b = torch.randn(10000, 10000)
start = time.time()
c = a @ b
print("CPU time:", time.time() - start)

# On CUDA
a_cuda = a.cuda()
b_cuda = b.cuda()
torch.cuda.synchronize()
start = time.time()
c_cuda = a_cuda @ b_cuda
torch.cuda.synchronize()
print("CUDA time:", time.time() - start)
