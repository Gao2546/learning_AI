# import pandas as pd
# from util.util import *
# from transformers import AutoTokenizer
# import random
# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from util.node import Transformer
# from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR

# class WarmupCosineScheduler:
#     def __init__(self, optimizer, warmup_steps, max_steps, base_lr):
#         self.optimizer = optimizer
#         self.warmup_steps = warmup_steps
#         self.max_steps = max_steps
#         self.base_lr = base_lr
#         self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
#         self.current_step = 0

#     def step(self):
#         if self.current_step < self.warmup_steps:
#             # Linear warmup
#             lr = self.base_lr * (self.current_step / self.warmup_steps)
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = lr
#         else:
#             # Cosine annealing
#             self.cosine_scheduler.step()

#         self.current_step += 1

# # data loader
# train_data = dataloadercustom()
# train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)

# src_vocab_size = train_data.token_size
# tgt_vocab_size = train_data.token_size
# d_model = 128*3
# num_heads = 6*1
# num_layers = 6*1
# d_ff = 2048//8
# max_seq_length = train_data.window_size
# dropout = 0.1

# transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
#                           num_heads, num_layers, d_ff, max_seq_length, dropout, device=0).to(device=0)

# # Generate random sample data
# # (batch_size, seq_length)
# src_data = torch.randint(
#     1, src_vocab_size, (64//8, max_seq_length)).to(device=0)
# # (batch_size, seq_length)
# tgt_data = torch.randint(
#     1, tgt_vocab_size, (64//8, max_seq_length)).to(device=0)

# criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=0)
# optimizer = optim.Adam(transformer.parameters(),
#                        lr=0.001, betas=(0.9, 0.95), eps=1e-9)

# # Learning rate scheduler
# warmup_steps = 500
# max_steps = 5000
# scheduler = WarmupCosineScheduler(optimizer, warmup_steps, max_steps, base_lr=0.001)
# # scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
# # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, )

# transformer.train()

# for epoch in tqdm(range(100)):
#     for question, answer_in, answer_out in tqdm(train_dataloader):
#         optimizer.zero_grad()
#         output = transformer(question, answer_in)
#         loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
#                          answer_out.contiguous().view(-1))
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

from util.model import Transformers, Transformer
import torch

# question = ["HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US",
#             "how a water pump works",
#             "how large were early jails",
#             "how old was sue lyon when she made lolita",
#             "how are antibodies used in",
#             "how old is alicia in 2009",
#             "how can i open a usda slaughterhouse",
#             "how deadly are brain tumors"]
question = ["# Write a program to check whether a number is prime or not",
            "# Write a program to find the factorial of a number",
            "# Write a program to check whether a number is positive, negative or zero",
            "# Write a python function to print whether a number is negative, positive or zero",
            "# write a program to find and print the largest among three numbers",
            "# Write a functin that returns the LCM of two input numbers",
            "# Write a function that returns the GCD of two input numbers",
            "# Write a program to check whether a number is a palindrome or not",
            "# Write a program to find the sum of natural numbers",]
# a = torch.zero((1,10))
# model = Transformers()
# model.train()

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

model = Transformer()
model.train()
# output = model.eval_model(question)
# for o in output:
#     print(o)
