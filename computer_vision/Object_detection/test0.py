import torch
import math
print(False * 10)
aa = torch.rand([10,10])
print(aa)
print((aa[:,:5] > 0.5)*10)
print(3/((80*80) + (40*40) + (20*20)))
print(10.112%1)
print(round(1.1))

tlcorner = torch.tensor([[[[i] for i in range(5)] for j in range(5)] for k in range(1)])
x_tlcorner = tlcorner
y_tlcorner = tlcorner.permute(0,2,1,3)
print(x_tlcorner)
print(y_tlcorner)