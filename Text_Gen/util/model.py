import pandas as pd
from util.util import *
from transformers import AutoTokenizer
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from util.node import Transformer
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import logging

logging.basicConfig(filename="output_04.log", level=logging.INFO)

class transformer:
    def __init__(self):
        self.save_model = True
        self.save_dir = "./model/transformer/"
        self.load_path = None#"model/transformer/transformer03_00300.pth"
        self.start_epoch = 0
        self.save_every_epoch = 1
        self.epochs = 10000
        self.batch_size = 64//8
        self.train_data = dataloadercustom()
        self.train_dataloader = DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True)
        self.pretrain_model_tokenizer_path = "./model/BPE_model/BPE_model_code_python03.pkl"
        self.device = 0
        self.sample_question = ["HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US",
                                "how a water pump works",
                                "how large were early jails",
                                "how old was sue lyon when she made lolita",
                                "how are antibodies used in",
                                "how old is alicia in 2009",
                                "how can i open a usda slaughterhouse",
                                "how deadly are brain tumors"]
        # self.sample_question = ["What are the differences between int, float, string, and bool in Python?",
        #                         "How do you check the data type of a variable?",
        #                         "Write a Python program to swap two variables.",
        #                         "Explain the difference between == and =.",
        #                         "Write a program that takes two numbers and prints their sum, difference, product, and quotient.",
        #                         "Write a Python program that checks if a given number is positive, negative, or zero.",
        #                         "Write a program to check if a year is a leap year or not.",
        #                         "What is the difference between if and elif?",
        #                         "Write a Python program to print numbers from 1 to 10 using a while loop.",
        #                         "Explain the difference between for and while loops.",
        #                         "Write a program to calculate the factorial of a number using a for loop.",
        #                         "What is the difference between a variable and an object?"]

        self.src_vocab_size = self.train_data.token_size
        self.tgt_vocab_size = self.train_data.token_size
        self.d_model = 128*6
        self.num_heads = 6*2
        self.num_layers = 6*2
        self.d_ff = 2048//2
        self.max_seq_length = self.train_data.window_size
        self.dropout = 0.1
        self.max_norm = 1.0

        self.tokenizer = BPE()
        self.tokenizer.load_pretrain(self.pretrain_model_tokenizer_path)
        self.transformer = Transformer(self.src_vocab_size, self.tgt_vocab_size, self.d_model,
                                  self.num_heads, self.num_layers, self.d_ff, self.max_seq_length, self.dropout, device=0).to(device=0)
        if self.load_path:
            self.load(self.load_path)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=0)
        self.optimizer = optim.AdamW(self.transformer.parameters(),
                               lr=0.0005, betas=(0.9, 0.95), eps=1e-9)

        # Learning rate scheduler
        self.warmup_steps = 318*500*8 #5%
        self.max_steps = 318*5000*8 #50%
        self.scheduler = WarmupCosineScheduler(self.optimizer, self.warmup_steps, self.max_steps, base_lr=0.0005, start_step=self.start_epoch*318*8)

    def train(self):
        self.transformer.train()
        for epoch in tqdm(range(self.start_epoch,self.epochs)):
            self.loss_epoch = []
            for question, answer_in, answer_out in tqdm(self.train_dataloader):
                self.optimizer.zero_grad()
                output = self.transformer(question, answer_in)
                loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                 answer_out.contiguous().view(-1))
                loss.backward()
                self.loss_epoch.append(loss.item())
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
                self.scheduler.step()
            if self.save and (((epoch + 1) % self.save_every_epoch) == 0):
                self.save(self.save_dir + f"transformer04_{epoch + 1:0=5}.pth")
                output_eval = self.eval_model(self.sample_question)
                logging.info(f"batch_eval : epoch {epoch}")
                for o in output_eval:
                    print(o)
                    logging.info(o)
                self.transformer.train()
            print(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}")
            logging.info(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}")

    def save(self,path):
        torch.save(self.transformer.state_dict(),path)

    def load(self,path):
        state_dict = torch.load(path)
        self.transformer.load_state_dict(state_dict=state_dict)

    def eval_model(self, questions):
        self.transformer.eval()
        output_list = []
        for question in questions:
            question_pre = [1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(question)) + [2, ]
            question_pre = torch.tensor(question_pre, device=self.device)
            question_pre = torch.nn.functional.pad(question_pre,(0,self.max_seq_length - len(question_pre)),"constant",0).unsqueeze(0)
            answer_output = torch.zeros((1,self.max_seq_length),device=self.device,dtype=torch.int32)
            answer_output[0,0] = 1
            answer_input = answer_output
            for seq_idx in range(self.max_seq_length - 1):
                if answer_output[0].cpu().tolist()[seq_idx] != 2:
                    answer_input[0,:seq_idx+1] = answer_output[0,:seq_idx+1]
                    answer_output = self.transformer(question_pre,answer_input)
                    answer_output = torch.argmax(torch.nn.functional.softmax(answer_output,dim=2),dim=2)
            output_list.append(question+ " :\n" + "".join(self.tokenizer.idx2token(answer_output[0,1:seq_idx].cpu().tolist())).replace("Ġ"," ").replace("Ċ","\n"))
        return output_list

            # self.answer_train.append([1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(a)) + [2, ])