import pandas as pd
from util.util import *
from transformers import AutoTokenizer
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from util.node import TransformerM , TransformersM , BertM
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import logging
import time
import math
import signal
import sys
import os

logging.basicConfig(filename="output_PythonSmall_04_3873data.log", level=logging.INFO)

class Transformers:
    def __init__(self):
        self.save_model = True
        self.save_dir = "./model/Transformers/"
        self.load_path = None#"model/transformer/transformer03_00300.pth"
        self.data_path = "./data/PythonCode500K/"
        self.tokenizer_path = "./model/BPE_model/tokenizer-bpe-10k.json"
        self.save_file = "Transformers_V01_128_384_6_6_1536_10K_MQtest2e-4.pth"
        self.start_epoch = 0
        self.save_every_epoch = 1
        self.epochs = 10000
        self.batch_size = 16
        self.max_seq_length = 128
        # self.train_data = dataloadercustom_Transformers()
        self.BPE_model = BPEs2(vocab_size=1024*5*2)
      
        self.BPE_model.train([self.data_path])
        self.BPE_model.load(self.tokenizer_path)
        self.train_data = data_loader3(self.data_path, new_tokenizer=self.BPE_model, max_len=self.max_seq_length)
        self.train_dataloader = DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True)
        # self.pretrain_model_tokenizer_path = "./model/BPE_model/BPE_model_code_python03.pkl"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_question = ["# Write a program to check whether a number is prime or not",
                                "# Write a program to find the factorial of a number",
                                "# Write a program to check whether a number is positive, negative or zero",
                                "# Write a python function to print whether a number is negative, positive or zero",
                                "# write a program to find and print the largest among three numbers",
                                "# Write a functin that returns the LCM of two input numbers",
                                "# Write a function that returns the GCD of two input numbers",
                                "# Write a program to check whether a number is a palindrome or not",
                                "# Write a program to find the sum of natural numbers",]
        # self.sample_question = ["HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US",
        #                         "how a water pump works",
        #                         "how large were early jails",
        #                         "how old was sue lyon when she made lolita",
        #                         "how are antibodies used in",
        #                         "how old is alicia in 2009",
        #                         "how can i open a usda slaughterhouse",
        #                         "how deadly are brain tumors"]
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

        self.src_vocab_size = self.BPE_model.tokenizer.get_vocab_size()
        self.tgt_vocab_size = self.BPE_model.tokenizer.get_vocab_size()
        self.d_model = 128*3
        self.num_heads = 6*1
        self.num_layers = 6*1
        self.d_ff = 128*3*4
        self.dropout = 0.1
        self.max_norm = 1.0

        # self.tokenizer = BPE()
        # self.tokenizer.load_pretrain(self.pretrain_model_tokenizer_path)
        self.Transformers = TransformersM(self.src_vocab_size, self.tgt_vocab_size, self.d_model,
                                  self.num_heads, self.num_layers, self.d_ff, self.max_seq_length, self.dropout, device=0).to(device=0)
        if self.load_path:
            # self.load(self.load_path)
            self.load_model_and_optimizer(self.load_path)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=0)
        self.optimizer = optim.AdamW(self.Transformers.parameters(),
                            #    lr=0.0005, betas=(0.9, 0.95), eps=1e-9)
                            lr=1e-4)

        # Learning rate scheduler
        self.warmup_steps = int(self.epochs*0.02*(math.ceil(len(self.train_data)/self.batch_size))) #5% 0.02
        self.max_steps = int(self.epochs*0.1*(math.ceil(len(self.train_data)/self.batch_size))) #50% 0.025
        self.scheduler = WarmupCosineScheduler(self.optimizer, self.warmup_steps, self.max_steps, base_lr=1e-4, start_step=self.start_epoch*318*8)

                # Count total parameters
        total_params = sum(p.numel() for p in self.Transformers.parameters())

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.Transformers.parameters() if p.requires_grad)
        # self.start_epoch = 146

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        print("Train data size: ", len(self.train_data))
        print("Batch size: ", self.batch_size)
        print("Total steps: ", math.ceil(len(self.train_data)/self.batch_size))
        print("Warmup steps: ", self.warmup_steps)
        print("Epochs: ", self.epochs)
        print("Example data: ")
        # for dd in self.train_data.get_sample()[0:self.train_data.amount_data:self.train_data.amount_data//10]: #get 10 sample data
        #     print(dd)
        for dd in self.train_data.get_sample(): #get 10 sample data
            print(dd)
            print("----------------------------------------------")


    def train(self):
        self.Transformers.train()
        for epoch in tqdm(range(self.start_epoch,self.epochs)):
            self.loss_epoch = []
            for question, answer_in, answer_out in tqdm(self.train_dataloader):
                self.optimizer.zero_grad()
                output = self.Transformers(question, answer_in)
                loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                 answer_out.contiguous().view(-1))
                loss.backward()
                self.loss_epoch.append(loss.item())
                torch.nn.utils.clip_grad_norm_(self.Transformers.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
                self.scheduler.step()
            if self.save and (((epoch + 1) % self.save_every_epoch) == 0):
                # self.save(self.save_dir + f"Transformers04_{epoch + 1:0=5}.pth")
                self.save_model_and_optimizer(self.save_dir + self.save_file)
                output_eval = self.eval_model(self.sample_question)
                logging.info(f"batch_eval : epoch {epoch}")
                for o in output_eval:
                    print(o)
                    logging.info(o)
                self.Transformers.train()
            print(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}")
            logging.info(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}")

    def save(self,path):
        torch.save(self.Transformers.state_dict(),path)

    def load(self,path):
        state_dict = torch.load(path)
        self.Transformers.load_state_dict(state_dict=state_dict)

    def save_model_and_optimizer(self, filepath):
        """
        Saves the state dictionaries of a model and its optimizer to a file.

        Parameters:
        model (torch.nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        filepath (str): The file path to save the state dictionaries.
        """
        checkpoint = {
            'model_state_dict': self.Transformers.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_schdule_step': self.schdule.current_step
        }
        torch.save(checkpoint, filepath)
        print(f"Model and optimizer state dictionaries saved to {filepath}")


    def load_model_and_optimizer(self, filepath, device='cpu'):
        """
        Loads the state dictionaries of a model and its optimizer from a file.

        Parameters:
        model (torch.nn.Module): The PyTorch model instance.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        filepath (str): The file path to load the state dictionaries from.
        device (str): The device to map the state dictionaries to ('cpu' or 'cuda').

        Returns:
        tuple: The model and optimizer with loaded states.
        """
        checkpoint = torch.load(filepath, map_location=device)
        if self.optimizer == None or self.schdule == None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.schdule.current_step = checkpoint['lr_schdule_step']
        print(f"Model and optimizer state dictionaries loaded from {filepath}")


    def eval_model(self, questions):
        self.Transformers.eval()
        output_list = []
        for question in questions:
            question_pre = self.BPE_model.tokenizer.encode(question)
            question_pre = torch.tensor(question_pre, device=self.device)
            question_pre = torch.nn.functional.pad(question_pre,(0,self.max_seq_length - len(question_pre)),"constant",0).unsqueeze(0)
            answer_output = torch.zeros((1,self.max_seq_length),device=self.device,dtype=torch.int32)
            answer_output[0,0] = 1
            answer_input = answer_output
            for seq_idx in range(self.max_seq_length - 1):
                if answer_output[0].cpu().tolist()[seq_idx] != 3:
                    answer_input[0,:seq_idx+1] = answer_output[0,:seq_idx+1]
                    answer_output = self.Transformers(question_pre,answer_input)
                    answer_output = torch.argmax(torch.nn.functional.softmax(answer_output,dim=2),dim=2)
            output_list.append("\n" + question+ " :\n" + self.BPE_model.decode_clean(answer_input[0,1:seq_idx].cpu().tolist()))
        return output_list

            # self.answer_train.append([1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(a)) + [2, ])

class Bert:
    def __init__(self):
        self.save_model = True
        self.save_dir = "./model/Bert/"
        self.load_path = None#"model/transformer/transformer03_00300.pth"
        self.start_epoch = 0
        self.save_every_epoch = 1
        self.epochs = 10000
        self.batch_size = 64//8
        self.train_data = dataloadercustom_Bert()
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
        self.Bert = BertM(self.src_vocab_size, self.tgt_vocab_size, self.d_model,
                                  self.num_heads, self.num_layers, self.d_ff, self.max_seq_length, self.dropout, device=0).to(device=0)
        if self.load_path:
            # self.load(self.load_path)
            self.load_model_and_optimizer(self.load_path,device=0)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=0)
        self.optimizer = optim.AdamW(self.Bert.parameters(),
                               lr=0.0005, betas=(0.9, 0.95), eps=1e-9)

        # Learning rate scheduler
        self.warmup_steps = 318*500*8 #5%
        self.max_steps = 318*5000*8 #50%
        self.scheduler = WarmupCosineScheduler(self.optimizer, self.warmup_steps, self.max_steps, base_lr=0.0005, start_step=self.start_epoch*318*8)

    def train(self):
        self.Bert.train()
        for epoch in tqdm(range(self.start_epoch,self.epochs)):
            self.loss_epoch = []
            for question, answer_out in tqdm(self.train_dataloader):
                self.optimizer.zero_grad()
                output = self.Bert(question)
                loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                 answer_out.contiguous().view(-1))
                loss.backward()
                self.loss_epoch.append(loss.item())
                torch.nn.utils.clip_grad_norm_(self.Bert.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
                self.scheduler.step()
            if self.save and (((epoch + 1) % self.save_every_epoch) == 0):
                # self.save(self.save_dir + f"Bert01_{epoch + 1:0=5}.pth")
                self.save_model_and_optimizer(self.save_dir + f"Bert01_{epoch + 1:0=5}.pth")
                output_eval = self.eval_model(self.sample_question)
                logging.info(f"batch_eval : epoch {epoch}")
                for o in output_eval:
                    print(o)
                    logging.info(o)
                self.Bert.train()
            print(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}")
            logging.info(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}")

    def save(self,path):
        torch.save(self.Bert.state_dict(),path)

    def load(self,path):
        state_dict = torch.load(path)
        self.Bert.load_state_dict(state_dict=state_dict)

    def save_model_and_optimizer(self, filepath):
        """
        Saves the state dictionaries of a model and its optimizer to a file.

        Parameters:
        model (torch.nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        filepath (str): The file path to save the state dictionaries.
        """
        checkpoint = {
            'model_state_dict': self.Transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_schdule_step': self.schdule.current_step
        }
        torch.save(checkpoint, filepath)
        print(f"Model and optimizer state dictionaries saved to {filepath}")


    def load_model_and_optimizer(self, filepath, device='cpu'):
        """
        Loads the state dictionaries of a model and its optimizer from a file.

        Parameters:
        model (torch.nn.Module): The PyTorch model instance.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        filepath (str): The file path to load the state dictionaries from.
        device (str): The device to map the state dictionaries to ('cpu' or 'cuda').

        Returns:
        tuple: The model and optimizer with loaded states.
        """
        checkpoint = torch.load(filepath, map_location=device)
        if self.optimizer == None or self.schdule == None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.schdule.current_step = checkpoint['lr_schdule_step']
        print(f"Model and optimizer state dictionaries loaded from {filepath}")


    def eval_model(self, questions):
        self.Bert.eval()
        output_list = []
        for question in questions:
            question_pre = [1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(question)) + [2, ]
            question_pre = torch.tensor(question_pre, device=self.device)
            question_pre = torch.nn.functional.pad(question_pre,(0,self.max_seq_length - len(question_pre)),"constant",0).unsqueeze(0)
            answer_output = self.Bert(question_pre)
            answer_output = torch.argmax(torch.nn.functional.softmax(answer_output,dim=2),dim=2)
            output_list = "".join(self.tokenizer.idx2token(answer_output[0,1:seq_idx].cpu().tolist())).replace("Ġ"," ").replace("Ċ","\n")
            # answer_output = torch.zeros((1,self.max_seq_length),device=self.device,dtype=torch.int32)
            # answer_output[0,0] = 1
            # answer_input = answer_output
            # for seq_idx in range(self.max_seq_length - 1):
                # if answer_output[0].cpu().tolist()[seq_idx] != 2:
                    # answer_input[0,:seq_idx+1] = answer_output[0,:seq_idx+1]
                    # answer_output = self.Bert(question_pre,answer_input)
                    # answer_output = torch.argmax(torch.nn.functional.softmax(answer_output,dim=2),dim=2)
            # output_list.append(question+ " :\n" + "".join(self.tokenizer.idx2token(answer_output[0,1:seq_idx].cpu().tolist())).replace("Ġ"," ").replace("Ċ","\n"))
        return output_list

# 2 1024*2
# 4 1024*4
# 3 1024*2 100 dataset
# class Transformer:
#     def __init__(self):
#         self.save_model = True
#         self.save_dir = "./model/Transformer/"
#         self.load_path = None#"./model/Transformer/Transformer_V03_10KA.pth" #DGood For Traning set
#         self.load_embedding_path = None
#         self.save_file = "Transformer_V03_10KA.pth"
#         #======================================================================================
#         # self.load_path = None#"./model/Transformer/Transformer_V01_10KC.pth" #DGood For Traning set
#         # self.save_file = "Transformer_VT01_10KA.pth"
        
#         self.start_epoch = 0
#         self.save_every_epoch = 1
#         self.epochs = 1000
#         self.batch_size = 64
#         self.train_data = dataloadercustom_Transformer(pretrain_model_tokenizer_path="./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl",qaaidx_path="./data/PythonCodeDataSmall_TextOnly/BPE_data/BPE_idx_V01_10K.pkl",amount_data=3873)
#         #========================================================================================
#         # self.train_data =  dataloadercustom_Transformer(pretrain_model_tokenizer_path="./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl",qaaidx_path="./data/PythonCodeDataSmall_TextOnly/BPE_data/BPE_idx_V01_10K.pkl",amount_data=10)
#         self.train_dataloader = DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True)
#         self.pretrain_model_tokenizer_path = "./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl"
#         self.device = 0
#         self.sample_question = ["# Write a program to check whether a number is prime or not",
#                                 "# Write a program to find the factorial of a number",
#                                 "# Write a program to check whether a number is positive, negative or zero",
#                                 "# Write a python function to print whether a number is negative, positive or zero",
#                                 "# write a program to find and print the largest among three numbers",
#                                 "# Write a functin that returns the LCM of two input numbers",
#                                 "# Write a function that returns the GCD of two input numbers",
#                                 "# Write a program to check whether a number is a palindrome or not",
#                                 "# Write a program to find the sum of natural numbers",
#                                 "# Write a Python Program to print the Sum of First N Natural Numbers",
#                                 "# Write a python program to print sum of number digits in List"]
#         # self.sample_question = [
#         #                         "num1 = 1.5\n",
#         #                         "def add_two_numbers(num1, num2):\n",
#         #                         "# write a program to find and print the largest among three numbers\n",
#         #                         "if (num1 >= num2) and (num1 >= num3):\n",
#         #                         "import os\n",
#         #                         "def two_power(terms):\n",
#         #                         "my_list = [1, 2, 3, 4, 5, 6]\n",
#         #                         "# Write a python function that returns the sum of n natural numbers\n"
#         #                         ]
#         # self.sample_question = ["What are the differences between int, float, string, and bool in Python?",
#         #                         "How do you check the data type of a variable?",
#         #                         "Write a Python program to swap two variables.",
#         #                         "Explain the difference between == and =.",
#         #                         "Write a program that takes two numbers and prints their sum, difference, product, and quotient.",
#         #                         "Write a Python program that checks if a given number is positive, negative, or zero.",
#         #                         "Write a program to check if a year is a leap year or not.",
#         #                         "What is the difference between if and elif?",
#         #                         "Write a Python program to print numbers from 1 to 10 using a while loop.",
#         #                         "Explain the difference between for and while loops.",
#         #                         "Write a program to calculate the factorial of a number using a for loop.",
#         #                         "What is the difference between a variable and an object?"]

#         self.src_vocab_size = self.train_data.token_size
#         self.tgt_vocab_size = self.train_data.token_size
#         self.d_model = 128*4 #6
#         self.num_heads = 6*1 #2
#         self.num_layers = 6*1 #2
#         self.d_ff = 512*2 #6
#         self.max_seq_length = self.train_data.window_size
#         self.dropout = 0.1
#         self.max_norm = 1.0

#         self.tokenizer = BPE()
#         self.tokenizer.load_pretrain(self.pretrain_model_tokenizer_path)
#         self.Transformer = TransformerM(self.src_vocab_size, self.tgt_vocab_size, self.d_model,
#                                   self.num_heads, self.num_layers, self.d_ff, self.max_seq_length, self.dropout, device=0).to(device=0)
#         # self.class_weights = torch.tensor([0, 1/5000, 1/5000] + [1/self.tokenizer.word_freqs[i] if self.tokenizer.word_freqs[i] != 0 else 0 for i in self.tokenizer.vocab[3:]],device=0)
#         self.class_weights = self.train_data.get_weight().to(device=0)
#         # self.criterion = nn.CrossEntropyLoss(ignore_index=0,weight=self.class_weights).to(device=0)
#         self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=0)
#         self.optimizer = optim.AdamW(self.Transformer.parameters(),
#                                lr=5e-5, betas=(0.9, 0.95), eps=1e-9) #lr is max learning rate lr=5e-5 //1e-5 1e-4 5e-6
                               

#         # Learning rate scheduler
#         self.warmup_steps = int(self.epochs*0.02*(math.ceil(len(self.train_data)/self.batch_size))) #5% 0.02
#         self.max_steps = int(self.epochs*0.1*(math.ceil(len(self.train_data)/self.batch_size))) #50% 0.025
#         self.scheduler = WarmupCosineScheduler(self.optimizer, self.warmup_steps, self.max_steps, base_lr=5e-5, start_step=None) #lr is max learning rate lr=5e-5 //1e-5 1e-4 5e-6

#         if self.load_path:
#             # self.load(self.load_path)
#             self.load_model_and_optimizer(self.load_path, only_model=False, device=0)
#             self.scheduler.cosine_scheduler.step(self.scheduler.current_step)
#         if self.load_embedding_path:
#             self.load_embedding(self.load_embedding_path)
#         else:
#             pass

#         # Count total parameters
#         total_params = sum(p.numel() for p in self.Transformer.parameters())

#         # Count trainable parameters
#         trainable_params = sum(p.numel() for p in self.Transformer.parameters() if p.requires_grad)
#         # self.start_epoch = 146

#         print(f"Total parameters: {total_params}")
#         print(f"Trainable parameters: {trainable_params}")

#         print("Train data size: ", len(self.train_data))
#         print("Batch size: ", self.batch_size)
#         print("Total steps: ", math.ceil(len(self.train_data)/self.batch_size))
#         print("Warmup steps: ", self.warmup_steps)
#         print("Epochs: ", self.epochs)
#         print("Example data: ")
#         for dd in self.train_data.get_sample()[0:self.train_data.amount_data:self.train_data.amount_data//10]: #get 10 sample data
#             print(dd)

#     def train(self):
#         self.Transformer.train()
#         for epoch in tqdm(range(self.start_epoch,self.epochs)):
#             self.loss_epoch = []
#             for answer_in, answer_out in tqdm(self.train_dataloader):
#                 signal.signal(signal.SIGINT, signal_handler)
#                 self.optimizer.zero_grad()
#                 output = self.Transformer(answer_in)
#                 # print(answer_in[0])
#                 # print(torch.argmax(torch.nn.functional.softmax(output[0],dim=1),dim=1))
#                 # print(answer_out[0])
#                 # print("=====================================")
#                 # time.sleep(3)
#                 loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
#                                  answer_out.contiguous().view(-1))
#                 loss.backward()
#                 self.loss_epoch.append(loss.item())
#                 torch.nn.utils.clip_grad_norm_(self.Transformer.parameters(), max_norm=self.max_norm)
#                 self.optimizer.step()
#                 self.scheduler.step()
#             if self.save and (((epoch + 1) % self.save_every_epoch) == 0):
#                 # self.save(self.save_dir + f"Transformer01_{epoch + 1:0=5}.pth")
#                 # self.save_model_and_optimizer(self.save_dir + f"Transformer01_{epoch + 1:0=5}.pth")
#                 self.save_model_and_optimizer(self.save_dir + f"{self.save_file}", epoch = epoch)
#                 # output_eval = self.eval_model(self.sample_question)
#                 logging.info(f"batch_eval : epoch {epoch}")
#                 # for o in output_eval:
#                 #     print(o)
#                 #     logging.info(o)
#                 # self.Transformer.train()
#             print(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}, lr: {self.optimizer.param_groups[0]['lr']}")
#             logging.info(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}, lr: {self.optimizer.param_groups[0]['lr']}")  

#             if (epoch + 1) % 5 == 0:
#                 self.save_model_and_optimizer(self.save_dir + "cpk/" + f"epoch_{epoch}_{self.save_file}", epoch = epoch)
#                 output_eval = self.eval_model(self.sample_question)
#                 for o in output_eval:
#                     print(o)
#                     logging.info(o)
#                     print("\n=====================================\n")
#                 self.Transformer.train()

#     def save(self,path):
#         torch.save(self.Transformer.state_dict(),path)
    
#     def load(self,path):
#         state_dict = torch.load(path)
#         self.Transformer.load_state_dict(state_dict=state_dict)

#     def save_model_and_optimizer(self, filepath, epoch):
#         """
#         Saves the state dictionaries of a model and its optimizer to a file.

#         Parameters:
#         model (torch.nn.Module): The PyTorch model.
#         optimizer (torch.optim.Optimizer): The optimizer for the model.
#         filepath (str): The file path to save the state dictionaries.
#         """
#         checkpoint = {
#             'model_state_dict': self.Transformer.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'lr_schdule_step': self.scheduler.current_step,
#             'current_epoch': epoch
#         }
#         torch.save(checkpoint, filepath)
#         print(f"Model and optimizer state dictionaries saved to {filepath}")


#     def load_model_and_optimizer(self, filepath, only_model, device='cpu'):
#         """
#         Loads the state dictionaries of a model and its optimizer from a file.

#         Parameters:
#         model (torch.nn.Module): The PyTorch model instance.
#         optimizer (torch.optim.Optimizer): The optimizer for the model.
#         filepath (str): The file path to load the state dictionaries from.
#         device (str): The device to map the state dictionaries to ('cpu' or 'cuda').

#         Returns:
#         tuple: The model and optimizer with loaded states.
#         """
#         if only_model:
#             checkpoint = torch.load(filepath)
#             self.Transformer.load_state_dict(checkpoint['model_state_dict'])
#             print(f"Model state dictionary loaded from {filepath}")
#             return 0
#         else:
#             checkpoint = torch.load(filepath)
#             if self.optimizer == None or self.scheduler == None:
#                 self.Transformer.load_state_dict(checkpoint['model_state_dict'])
#             else:
#                 self.Transformer.load_state_dict(checkpoint['model_state_dict'])
#                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#                 self.scheduler.current_step = checkpoint['lr_schdule_step']
#                 self.start_epoch = checkpoint['current_epoch'] + 1
#             print(f"Model and optimizer state dictionaries loaded from {filepath}")
#             # return self.start_epoch

#     def load_embedding(self, path):
#         state_dict = torch.load(path)
#         self.Transformer.decoder_embedding.weight.data.copy_(state_dict['model_state_dict']['decoder_embedding.weight'])
#         print(f"Model and optimizer state dictionaries loaded from {path}")

#     def eval_model(self, questions):
#         self.Transformer.eval()
#         output_list = []
#         for question in questions:
#             answer_output = [1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(question))
#             start_seq = len(answer_output) - 1
#             # print(start_seq)
#             answer_output = torch.tensor(answer_output, device=self.device)
#             answer_output = torch.nn.functional.pad(answer_output,(0,self.max_seq_length - len(answer_output)),"constant",0).unsqueeze(0)
#             # answer_output = torch.zeros((1,self.max_seq_length),device=self.device,dtype=torch.int32)
#             # answer_output[0,0] = 1
#             answer_input = answer_output.clone()
#             for seq_idx in range(start_seq,self.max_seq_length - 1):
#                 if answer_output[0].clone().cpu().tolist()[seq_idx] != 2:
#                     answer_input[0,seq_idx] = answer_output.clone()[0,seq_idx]
#                     answer_output = self.Transformer(answer_input)
#                     answer_output = torch.argmax(torch.nn.functional.softmax(answer_output,dim=2),dim=2)
#                     # print(f"seq_idx: {seq_idx}, answer_output: {answer_output}")
#                 else:
#                     break
#             answer_input[0,seq_idx] = answer_output.clone()[0,seq_idx]
#             output_list.append("\n" + question+ " :\n" + "".join(self.tokenizer.idx2token(answer_input[0,1:seq_idx].cpu().tolist())).replace("Ġ"," ").replace("Ċ","\n"))
#         return output_list


class Transformer:
    def __init__(self): #loss == 0.02 0.006
        self.save_model = True
        self.save_dir = "./model/Transformer/"
        self.load_path = "./model/Transformer/Transformer_V01_128_512_12_12_2048_10k_MQfulldataCKP1.pth" #DGood For Traning set
        self.load_embedding_path = None#"./model/Transformer/embedding_model.pth"
        self.data_path = "./data/PythonCodeDataSmall_TextOnly/Python_code_data3ex.txt"
        self.data_path_full = "./data/PythonCodeDataSmall_TextOnly/Python_code_data.txt"
        self.tokenizer_path = "./model/BPE_model/tokenizer-bpe-5k.json"
        self.save_file = "Transformer_V01_128_512_12_12_2048_10K_MQtest2e-4.pth"
        #======================================================================================
        # self.load_path = None#"./model/Transformer/Transformer_V01_10KC.pth" #DGood For Traning set
        # self.save_file = "Transformer_VT01_10KA.pth"
        
        self.start_epoch = 0
        self.save_every_epoch = 100
        self.epochs = 10000
        self.batch_size = 16*1
        self.max_seq_length = 128 #512
        # self.train_data = dataloadercustom_Transformer(pretrain_model_tokenizer_path="./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl",qaaidx_path="./data/PythonCodeDataSmall_TextOnly/BPE_data/BPE_idx_V01_10K.pkl",amount_data=3873)
        self.BPE_model = BPEs(vocab_size=1024*5*2)
      
        self.BPE_model.train([self.data_path_full])
        self.BPE_model.load(self.tokenizer_path)
        self.train_data = data_loader(self.data_path, new_tokenizer=self.BPE_model, max_len=self.max_seq_length)
        #========================================================================================
        # self.train_data =  dataloadercustom_Transformer(pretrain_model_tokenizer_path="./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl",qaaidx_path="./data/PythonCodeDataSmall_TextOnly/BPE_data/BPE_idx_V01_10K.pkl",amount_data=10)
        self.train_dataloader = DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True)
        # self.pretrain_model_tokenizer_path = "./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_question = ["# write a program to find and print the largest among three numbers",
                                "# Write a program to check whether a number is prime or not",
                                # "# Write a program to check whether a number is prime or not",
                                "# Write a python function to print whether a number is negative, positive or zero",
                                "# Write a program to print the sum of squares of first n natural numbers",
                                "# Write a program to find the sum of digits of a number",
                                "# Write a program to find the factorial of a number",
                                "# Write a program to check whether a number is positive, negative or zero",
                                "# Write a python function to print whether a number is negative, positive or zero",
                                # "# write a program to find and print the largest among three numbers",
                                # "# Write a functin that returns the LCM of two input numbers",
                                # "# Write a function that returns the GCD of two input numbers",
                                # "# Write a program to check whether a number is a palindrome or not",
                                # "# Write a program to find the sum of natural numbers",
                                # "# Write a Python Program to print the Sum of First N Natural Numbers",
                                "# Write a python program to print sum of number digits in List"]

        # self.src_vocab_size = self.train_data.token_size
        # self.tgt_vocab_size = self.train_data.token_size
        self.src_vocab_size = 1024*5*2
        self.tgt_vocab_size = 1024*5*2
        self.d_model = 128*6 #6
        self.num_heads = 6*2 #6*1 #2
        self.num_layers = 6*2 #2
        self.d_ff = 128*4*6#512*2 #6
        # self.max_seq_length = self.train_data.window_size
        self.dropout = 0.1
        self.max_norm = 1.0

        # self.tokenizer = BPE()
        # self.tokenizer.load_pretrain(self.pretrain_model_tokenizer_path)
        self.Transformer = TransformerM(self.src_vocab_size, self.tgt_vocab_size, self.d_model,
                                  self.num_heads, self.num_layers, self.d_ff, self.max_seq_length, self.dropout, device=0).to(device=0)
        # self.Transformer.decoder_embedding.requires_grad_ = False
        # self.class_weights = torch.tensor([0, 1/5000, 1/5000] + [1/self.tokenizer.word_freqs[i] if self.tokenizer.word_freqs[i] != 0 else 0 for i in self.tokenizer.vocab[3:]],device=0)
        # self.class_weights = self.train_data.get_weight().to(device=0)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=0,weight=self.class_weights).to(device=0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=0)
        self.optimizer = optim.AdamW(self.Transformer.parameters(),
                                    #  lr=2e-4)
                               lr=5e-5, betas=(0.9, 0.95), eps=1e-9) #lr is max learning rate lr=5e-5 //1e-5 1e-4 5e-6
                               

        # Learning rate scheduler
        self.warmup_steps = int(self.epochs*0.02*(math.ceil(len(self.train_data)/self.batch_size))) #5% 0.02
        self.max_steps = int(self.epochs*0.1*(math.ceil(len(self.train_data)/self.batch_size))) #50% 0.025
        self.scheduler = WarmupCosineScheduler(self.optimizer, self.warmup_steps, self.max_steps, base_lr=5e-5, start_step=None) #lr is max learning rate lr=5e-5 //1e-5 1e-4 5e-6

        if self.load_path:
            # self.load(self.load_path)
            self.load_model_and_optimizer(self.load_path, only_model=False, device=0)
            self.scheduler.cosine_scheduler.step(self.scheduler.current_step)
        if self.load_embedding_path:
            self.load_embedding(self.load_embedding_path)
        else:
            pass

        # Count total parameters
        total_params = sum(p.numel() for p in self.Transformer.parameters())

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.Transformer.parameters() if p.requires_grad)
        # self.start_epoch = 146

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        print("Train data size: ", len(self.train_data))
        print("Batch size: ", self.batch_size)
        print("Total steps: ", math.ceil(len(self.train_data)/self.batch_size))
        print("Warmup steps: ", self.warmup_steps)
        print("Epochs: ", self.epochs)
        print("Example data: ")
        # for dd in self.train_data.get_sample()[0:self.train_data.amount_data:self.train_data.amount_data//10]: #get 10 sample data
        #     print(dd)
        for dd in self.train_data.get_sample(): #get 10 sample data
            print(dd)
            print("----------------------------------------------")

    def train(self):
        self.Transformer.train()
        for epoch in tqdm(range(self.start_epoch,self.epochs)):
            self.loss_epoch = []
            for answer_in, answer_out in tqdm(self.train_dataloader):
                signal.signal(signal.SIGINT, signal_handler)
                self.optimizer.zero_grad()
                output = self.Transformer(answer_in)
                # print(answer_in[0])
                # print(torch.argmax(torch.nn.functional.softmax(output[0],dim=1),dim=1))
                # print(answer_out[0])
                # print("=====================================")
                # time.sleep(3)
                loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                 answer_out.contiguous().view(-1))
                loss.backward()
                self.loss_epoch.append(loss.item())
                torch.nn.utils.clip_grad_norm_(self.Transformer.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
                self.scheduler.step()
            if self.save and (((epoch + 1) % self.save_every_epoch) == 0):
                # self.save(self.save_dir + f"Transformer01_{epoch + 1:0=5}.pth")
                # self.save_model_and_optimizer(self.save_dir + f"Transformer01_{epoch + 1:0=5}.pth")
                self.save_model_and_optimizer(self.save_dir + f"{self.save_file}", epoch = epoch)
                # output_eval = self.eval_model(self.sample_question)
                logging.info(f"batch_eval : epoch {epoch}")
                # for o in output_eval:
                #     print(o)
                #     logging.info(o)
                # self.Transformer.train()
            print(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}, lr: {self.optimizer.param_groups[0]['lr']}")
            logging.info(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}, lr: {self.optimizer.param_groups[0]['lr']}")  

            if (epoch + 1) % 100 == 0:
                # self.save_model_and_optimizer(self.save_dir + "cpk/" + f"epoch_{epoch}_{self.save_file}", epoch = epoch)
                output_eval = self.eval_model(self.sample_question)
                for o in output_eval:
                    print(o)
                    logging.info(o)
                    print("\n=====================================\n")
                self.Transformer.train()

    def save(self,path):
        torch.save(self.Transformer.state_dict(),path)
    
    def load(self,path):
        state_dict = torch.load(path)
        self.Transformer.load_state_dict(state_dict=state_dict)

    def save_model_and_optimizer(self, filepath, epoch):
        """
        Saves the state dictionaries of a model and its optimizer to a file.

        Parameters:
        model (torch.nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        filepath (str): The file path to save the state dictionaries.
        """
        checkpoint = {
            'model_state_dict': self.Transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_schdule_step': self.scheduler.current_step,
            'current_epoch': epoch
        }
        torch.save(checkpoint, filepath)
        print(f"Model and optimizer state dictionaries saved to {filepath}")


    def load_model_and_optimizer(self, filepath, only_model, device='cpu'):
        """
        Loads the state dictionaries of a model and its optimizer from a file.

        Parameters:
        model (torch.nn.Module): The PyTorch model instance.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        filepath (str): The file path to load the state dictionaries from.
        device (str): The device to map the state dictionaries to ('cpu' or 'cuda').

        Returns:
        tuple: The model and optimizer with loaded states.
        """
        if only_model:
            checkpoint = torch.load(filepath)
            self.Transformer.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model state dictionary loaded from {filepath}")
            return 0
        else:
            checkpoint = torch.load(filepath)
            if self.optimizer == None or self.scheduler == None:
                self.Transformer.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.Transformer.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.current_step = checkpoint['lr_schdule_step']
                self.start_epoch = checkpoint['current_epoch'] + 1
            print(f"Model and optimizer state dictionaries loaded from {filepath}")
            # return self.start_epoch

    def load_embedding(self, path):
        state_dict = torch.load(path)
        # self.Transformer.decoder_embedding.weight.data.copy_(state_dict['model_state_dict']['decoder_embedding.weight'])
        self.Transformer.decoder_embedding.load_state_dict(state_dict)
        print(f"Model and optimizer state dictionaries loaded from {path}")

    def eval_model(self, questions):
        self.Transformer.eval()
        output_list = []
        for question in questions:
            # answer_output = [1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(question))
            answer_output = [1, ] + self.BPE_model.tokenizer.encode(question).ids
            start_seq = len(answer_output) - 1
            # print(start_seq)
            answer_output = torch.tensor(answer_output, device=self.device)
            answer_output = torch.nn.functional.pad(answer_output,(0,self.max_seq_length - len(answer_output)),"constant",0).unsqueeze(0)
            # answer_output = torch.zeros((1,self.max_seq_length),device=self.device,dtype=torch.int32)
            # answer_output[0,0] = 1
            answer_input = answer_output.clone()
            seq_idx = 0
            for seq_idx in range(start_seq,self.max_seq_length - 1):
                if answer_output[0].clone().cpu().tolist()[seq_idx] != 3:
                    answer_input[0,seq_idx] = answer_output.clone()[0,seq_idx]
                    answer_output = self.Transformer(answer_input)
                    answer_output = torch.argmax(torch.nn.functional.softmax(answer_output,dim=2),dim=2)
                    # print(f"seq_idx: {seq_idx}, answer_output: {answer_output}")
                else:
                    break
            answer_input[0,seq_idx] = answer_output.clone()[0,seq_idx]
            # output_list.append("\n" + question+ " :\n" + "".join(self.tokenizer.idx2token(answer_input[0,1:seq_idx].cpu().tolist())).replace("Ġ"," ").replace("Ċ","\n"))
            output_list.append("\n" + question+ " :\n" + self.BPE_model.decode_clean(answer_input[0,1:seq_idx].cpu().tolist()))
        return output_list


    
    # def eval_model(self, questions):
    #     self.Transformer.eval()
    #     torch.no_grad()
    #     output_list = []
    #     for question in questions:
    #         answer_output = [1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(question))
    #         start_seq = len(answer_output)
    #         answer_output = torch.tensor(answer_output, device=self.device)
    #         answer_output = torch.nn.functional.pad(answer_output, (0, self.max_seq_length - len(answer_output)), "constant", 0).unsqueeze(0)
    #         answer_input = answer_output.clone()

    #         print(f"Initial answer_output: {answer_output}")
    #         print(f"start_seq: {start_seq}, max_seq_length: {self.max_seq_length}")

    #         for seq_idx in range(start_seq, self.max_seq_length - 1):
    #             if answer_output[0].clone().cpu().tolist()[seq_idx] != 2:
    #                 answer_input[0, seq_idx] = answer_output[0, seq_idx]
    #                 answer_output = self.Transformer(answer_input)
    #                 answer_output = torch.argmax(torch.nn.functional.softmax(answer_output, dim=2), dim=2)
    #                 print(f"seq_idx: {seq_idx}, answer_output: {answer_output}")

    #                 # Check if indices are within valid range
    #                 if torch.any(answer_output >= self.tgt_vocab_size):
    #                     raise ValueError(f"Index out of range in answer_output: {answer_output}")

    #         output_list.append(question + " :\n" + "".join(self.tokenizer.idx2token(answer_output[0, 1:seq_idx].cpu().tolist())).replace("Ġ", " ").replace("Ċ", "\n"))
    #     return output_list
def signal_handler(sig, frame):
    print("Training interrupted by user")
    # Clear all data in GPU
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    sys.exit(0)
