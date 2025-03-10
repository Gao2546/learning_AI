import torch
import torch.nn as nn
import torch.optim as optim
# import torch.utils.data as data
from util.node import Transformer
from torch.optim.lr_scheduler import StepLR
# import math
# import copy


src_vocab_size = 50000
tgt_vocab_size = 50000
d_model = 512//1
num_heads = 8
num_layers = 6
d_ff = 2048//2
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                          num_heads, num_layers, d_ff, max_seq_length, dropout, device=0).to(device=0)

# Generate random sample data
# (batch_size, seq_length)
src_data = torch.randint(
    1, src_vocab_size, (64//8, max_seq_length)).to(device=0)
# (batch_size, seq_length)
tgt_data = torch.randint(
    1, tgt_vocab_size, (64//8, max_seq_length)).to(device=0)

criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=0)
optimizer = optim.Adam(transformer.parameters(),
                       lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                     tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
