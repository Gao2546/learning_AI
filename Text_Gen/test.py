# import numpy as np
import transformers
# import torch
# a = torch.rand((64, 100))
# print(a.size())
# b = (a != 0).unsqueeze(1).unsqueeze(2)
# print(b)
# print(b.size())
# c = a.unsqueeze(1).unsqueeze(3)
# print(c.size())
# src_vocab_size = 500
# max_seq_length = 100
# tgt_data = torch.randint(
#     1, src_vocab_size, (64, max_seq_length)).to(device="cpu")
# tgt_data = tgt_data.unsqueeze(1).unsqueeze(3)
# print(tgt_data.size())
# seq_length = tgt_data.size(2)
# nopeak_mask = (
#     1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
# print(nopeak_mask.size())
# print(nopeak_mask)
# print((tgt_data & nopeak_mask).size())
