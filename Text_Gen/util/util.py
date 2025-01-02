from collections import defaultdict, Counter
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import torch
import random
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

tokenizer = AutoTokenizer.from_pretrained("gpt2")


class BPE:
    def __init__(self):
        self.vocab = []
        self.merges = {}
        self.splits = {}
        self.word_freqs = defaultdict(int)

    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                    i += 1
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def train(self, corpus, vocab_size):
        print("count word freq")
        for text in tqdm(corpus):
            words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1

        alphabet = []
        print("find alphabet")
        for word in tqdm(self.word_freqs.keys()):
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        self.vocab = ["<|pad|>", "<|startoftext|>", "<|endoftext|>"] + alphabet.copy()
        self.splits = {word: [c for c in word]
                       for word in self.word_freqs.keys()}
        with tqdm(total=vocab_size) as pbar:
            pbar.update(len(self.vocab))
            while (len(self.vocab) < vocab_size):
                pair_freqs = self.compute_pair_freqs()
                if len(pair_freqs) == 0:
                    break
                best_pair = ""
                max_freq = None
                # best_pair = max(pair_freqs, key=pair_freqs.get)
                # max_freq = pair_freqs[best_pair]
                for pair, freq in pair_freqs.items():
                    if max_freq is None or max_freq < freq:
                        best_pair = pair
                        max_freq = freq
                self.splits = self.merge_pair(
                    *best_pair)
                self.merges[best_pair] = best_pair[0] + best_pair[1]
                self.vocab.append(best_pair[0] + best_pair[1])
                pbar.update(1)

    def tokenize(self, text):
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(
            text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits = [[l for l in word] for word in pre_tokenized_text]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])
    
    def token2idx(self, token):
        return [self.vocab.index(t) for t in token]
    
    def idx2token(self, idx):
        return [self.vocab[ids] for ids in idx]

    def decode(self, tokens):
        sentence = "".join(tokens).replace("Ġ", " ")
        return sentence

    def load_pretrain(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vocab = data["vocab"]
        self.merges = data["merges"]
        self.splits = data["splits"]
        self.word_freqs = data["word_freqs"]

    def save_model(self, path):
        data = {"vocab": self.vocab,
                "merges": self.merges,
                "splits": self.splits,
                "word_freqs": self.word_freqs}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def add_vocabs(self, new_vocabs: list):
        self.vocab = new_vocabs + self.vocab


class dataloadercustom_Transformers(Dataset):

    def __init__(self,
                 token_size: int = 1024*5,
                 window_size: int = 64*2,
                 pretrain_model_tokenizer_path: str = "./model/BPE_model/BPE_model_code_python03.pkl",
                #  data_path: str = "./data/question_and_answer_no_code01/Dataset_Python_Question_Answer.csv",
                 data_path: str = "data/WikiQACorpus/WikiQA-train.tsv",
                 tokenizer_model: str = "bpe",
                 device: int = 0,
                 qaaidx_path: str = "./data/question_and_answer_no_code01/BPE_model_code_python01/qaaidx_token03.pkl",
                 qaaidx_save: str = True):
        self.device = device
        self.data_path = data_path
        self.window_size = window_size
        self.qaaidx_path = qaaidx_path
        self.question = []
        self.answer = []
        self.question_train = []
        self.answer_train = []
        # data = pd.read_csv("./data/question_and_answer_no_code01/Dataset_Python_Question_Answer.csv", chunksize=10000)
        data = pd.read_csv("data/WikiQACorpus/WikiQA-train.tsv", chunksize=10000,sep="\t")
        if os.path.isfile(self.qaaidx_path):
            if tokenizer_model == "bpe":
                self.tokenizer = BPE()
                self.tokenizer.load_pretrain(pretrain_model_tokenizer_path)
                with open(self.qaaidx_path,"rb") as f:
                    self.question_train,self.answer_train = pickle.load(f)
        else:
            for d in tqdm(data):
                # self.question += d["Question"].tolist()
                # ans = d["Answer"].map(lambda x: "\n".join(eval(x))).tolist()
                # self.answer += ans
                self.question += d["Question"].tolist()
                self.answer += d["Sentence"].tolist()
            del data
            if tokenizer_model == "bpe":
                self.tokenizer = BPE()
                self.tokenizer.load_pretrain(pretrain_model_tokenizer_path)
                for q,a in tqdm(zip(self.question,self.answer)):
                    self.question_train.append([1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(q)) + [2, ])
                    self.answer_train.append([1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(a)) + [2, ])
                if (self.qaaidx_path != None) and (self.qaaidx_path[-4:] == ".pkl") and qaaidx_save:
                    with open(self.qaaidx_path,"wb") as f:
                        pickle.dump([self.question_train,self.answer_train],f)
        self.token_size = len(self.tokenizer.vocab)


    def __len__(self):
        #return sum([len(data) for data in pd.read_csv(self.data_path,chunksize=10)])
        return len(self.question_train)

    def __getitem__(self, index):
        question_train = torch.tensor(self.question_train[index],device=self.device)
        question_train = torch.nn.functional.pad(question_train,(0,self.window_size - len(question_train)),"constant",0)
        answer_train = self.answer_train[index]
        r_posi = random.randrange(1,len(answer_train) - 1,1)
        answer_train_in = torch.tensor(answer_train[0:r_posi],device=self.device)
        answer_train_out = torch.tensor(answer_train[0:r_posi+1],device=self.device)
        answer_train_in = torch.nn.functional.pad(answer_train_in,(0,self.window_size - len(answer_train_in)),"constant",0)
        answer_train_out = torch.nn.functional.pad(answer_train_out,(0,self.window_size - len(answer_train_out)),"constant",0)
        return question_train,answer_train_in,answer_train_out

    def get_vocab(self):
        return self.tokenizer.vocab
    
class dataloadercustom_Bert(Dataset):

    def __init__(self,
                 token_size: int = 1024*5,
                 window_size: int = 64*2,
                 pretrain_model_tokenizer_path: str = "./model/BPE_model/BPE_model_code_python03.pkl",
                #  data_path: str = "./data/question_and_answer_no_code01/Dataset_Python_Question_Answer.csv",
                 data_path: str = "data/WikiQACorpus/WikiQA-train.tsv",
                 tokenizer_model: str = "bpe",
                 device: int = 0,
                 qaaidx_path: str = "./data/question_and_answer_no_code01/BPE_model_code_python01/qaaidx_token03.pkl",
                 qaaidx_save: str = True):
        self.device = device
        self.data_path = data_path
        self.window_size = window_size
        self.qaaidx_path = qaaidx_path
        self.question = []
        self.answer = []
        self.question_train = []
        self.answer_train = []
        # data = pd.read_csv("./data/question_and_answer_no_code01/Dataset_Python_Question_Answer.csv", chunksize=10000)
        data = pd.read_csv("data/WikiQACorpus/WikiQA-train.tsv", chunksize=10000,sep="\t")
        if os.path.isfile(self.qaaidx_path):
            if tokenizer_model == "bpe":
                self.tokenizer = BPE()
                self.tokenizer.load_pretrain(pretrain_model_tokenizer_path)
                with open(self.qaaidx_path,"rb") as f:
                    self.question_train,self.answer_train = pickle.load(f)
        else:
            for d in tqdm(data):
                # self.question += d["Question"].tolist()
                # ans = d["Answer"].map(lambda x: "\n".join(eval(x))).tolist()
                # self.answer += ans
                self.question += d["Question"].tolist()
                self.answer += d["Sentence"].tolist()
            del data
            if tokenizer_model == "bpe":
                self.tokenizer = BPE()
                self.tokenizer.load_pretrain(pretrain_model_tokenizer_path)
                for q,a in tqdm(zip(self.question,self.answer)):
                    self.question_train.append([1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(q)) + [2, ])
                    self.answer_train.append([1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(a)) + [2, ])
                if (self.qaaidx_path != None) and (self.qaaidx_path[-4:] == ".pkl") and qaaidx_save:
                    with open(self.qaaidx_path,"wb") as f:
                        pickle.dump([self.question_train,self.answer_train],f)
        self.token_size = len(self.tokenizer.vocab)


    def __len__(self):
        #return sum([len(data) for data in pd.read_csv(self.data_path,chunksize=10)])
        return len(self.question_train)

    def __getitem__(self, index):
        question_train = torch.tensor(self.question_train[index],device=self.device)
        question_train = torch.nn.functional.pad(question_train,(0,self.window_size - len(question_train)),"constant",0)
        answer_train = self.answer_train[index]
        r_posi = random.randrange(1,len(answer_train) - 1,1)
        answer_train_in = torch.tensor(answer_train[0:r_posi],device=self.device)
        answer_train_out = torch.tensor(answer_train[0:r_posi+1],device=self.device)
        answer_train_in = torch.nn.functional.pad(answer_train_in,(0,self.window_size - len(answer_train_in)),"constant",0)
        answer_train_out = torch.nn.functional.pad(answer_train_out,(0,self.window_size - len(answer_train_out)),"constant",0)
        return question_train,answer_train_in,answer_train_out

    def get_vocab(self):
        return self.tokenizer.vocab
    
# class dataloadercustom_Transformer(Dataset):

#     def __init__(self,
#                  token_size: int = 1024*2,
#                  window_size: int = 64*2,
#                  pretrain_model_tokenizer_path: str = "./model/BPE_model/BPE_model_code_python_small_text03.pkl",
#                 #  data_path: str = "./data/question_and_answer_no_code01/Dataset_Python_Question_Answer.csv",
#                  data_path: str = "/home/athip/psu/learning_AI/Text_Gen/data/PythonCodeDataSmall_TextOnly/Python_code_data.txt",
#                  tokenizer_model: str = "bpe",
#                  device: int = 0,
#                  qaaidx_path: str = "/home/athip/psu/learning_AI/Text_Gen/data/PythonCodeDataSmall_TextOnly/BPE_data/BPE_idx03.pkl",
#                  qaaidx_save: str = True):
#         self.device = device
#         self.data_path = data_path
#         self.window_size = window_size
#         self.qaaidx_path = qaaidx_path
#         self.question = []
#         self.answer = []
#         self.question_train = []
#         self.answer_train = []
#         # data = pd.read_csv("./data/question_and_answer_no_code01/Dataset_Python_Question_Answer.csv", chunksize=10000)
#         # data = pd.read_csv("data/WikiQACorpus/WikiQA-train.tsv", chunksize=10000,sep="\t")
#         with open("/home/athip/psu/learning_AI/Text_Gen/data/PythonCodeDataSmall_TextOnly/Python_code_data.txt","r") as f:
#             data = f.read(-1)
#             data = data.split("\n# ")
#             data = [data[0].strip("\n")] + [("# " + c).strip("\n") for c in data[1:] if len(c) >= 150]
#         if os.path.isfile(self.qaaidx_path):
#             if tokenizer_model == "bpe":
#                 self.tokenizer = BPE()
#                 self.tokenizer.load_pretrain(pretrain_model_tokenizer_path)
#                 with open(self.qaaidx_path,"rb") as f:
#                     self.answer_train = pickle.load(f)
#         else:
#             # for d in tqdm(data):
#                 # self.question += d["Question"].tolist()
#                 # ans = d["Answer"].map(lambda x: "\n".join(eval(x))).tolist()
#                 # self.answer += ans
#                 # self.question += d["Question"].tolist()
#                 # self.answer += d["Sentence"].tolist()
#                 # self.answer += 
#             self.answer = data
#             del data
#             if tokenizer_model == "bpe":
#                 self.tokenizer = BPE()
#                 self.tokenizer.load_pretrain(pretrain_model_tokenizer_path)
#                 for a in tqdm(self.answer):
#                     self.answer_train.append([1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(a)) + [2, ])
#                     # self.answer_train.append(self.tokenizer.token2idx(self.tokenizer.tokenize(a)) + [2, ])
#                 if (self.qaaidx_path != None) and (self.qaaidx_path[-4:] == ".pkl") and qaaidx_save:
#                     with open(self.qaaidx_path,"wb") as f:
#                         pickle.dump(self.answer_train,f)
#         self.token_size = len(self.tokenizer.vocab)


#     def __len__(self):
#         #return sum([len(data) for data in pd.read_csv(self.data_path,chunksize=10)])
#         return len(self.answer_train)

#     def __getitem__(self, index):
#         # question_train = torch.tensor(self.question_train[index],device=self.device)
#         # question_train = torch.nn.functional.pad(question_train,(0,self.window_size - len(question_train)),"constant",0)
#         answer_train = self.answer_train[index]
#         s_posi = random.randrange(0, len(answer_train) - 2, 1)
#         r_posi = random.randrange(s_posi + 1, len(answer_train) - 1, 1)
#         answer_train_in = torch.tensor(answer_train[s_posi:r_posi],device=self.device)
#         answer_train_out = torch.tensor(answer_train[s_posi:r_posi+1],device=self.device)
#         answer_train_in = torch.nn.functional.pad(answer_train_in,(0,self.window_size - len(answer_train_in)),"constant",0)
#         answer_train_out = torch.nn.functional.pad(answer_train_out,(0,self.window_size - len(answer_train_out)),"constant",0)
#         # print(answer_train_in)
#         # print(answer_train_out)
#         return answer_train_in,answer_train_out

#     def get_vocab(self):
#         return self.tokenizer.vocab
#     def get_weight(self):
#         self.answer_feq = Counter([t for a in self.answer_train for t in a])
#         all_value = sum([len(Tvalues) for Tvalues in self.answer_train])
#         min_value = min(self.answer_feq.values())
#         max_ratio = all_value/min_value
#         sum_value = sum(self.answer_feq.values())
#         self.weight = torch.tensor([((all_value/(self.answer_feq[t] + 1))/max_ratio)**(1/1.0) for t in range(len(self.tokenizer.vocab))]) #2.71828182846
#         # print(self.weight.max())
#         # time.sleep(100)
#         # print(self.answer_feq)
#         # print(self.weight)
#         # print(self.weight.min())
#         # print(all_value/(self.answer_feq[103] + 1))
#         # print(all_value)
#         # print(self.tokenizer.vocab)
#         # time.sleep(100)
#         return self.weight
    

class dataloadercustom_Transformer(Dataset):

    def __init__(self,
                 token_size: int = 1024*2,
                 window_size: int = 64*2,
                 pretrain_model_tokenizer_path: str = "./model/BPE_model/BPE_model_code_python_small_text03.pkl",
                #  data_path: str = "./data/question_and_answer_no_code01/Dataset_Python_Question_Answer.csv",
                 data_path: str = "/home/athip/psu/learning_AI/Text_Gen/data/PythonCodeDataSmall_TextOnly/Python_code_data.txt",
                 tokenizer_model: str = "bpe",
                 device: int = 0,
                 qaaidx_path: str = "/home/athip/psu/learning_AI/Text_Gen/data/PythonCodeDataSmall_TextOnly/BPE_data/BPE_idx03.pkl",
                 qaaidx_save: str = True,
                 amount_data: int = 100):
        self.device = device
        self.data_path = data_path
        self.window_size = window_size
        self.qaaidx_path = qaaidx_path
        self.question = []
        self.answer = []
        self.question_train = []
        self.answer_train_in = []
        self.answer_train_out = []
        self.answer_all = []
        self.amount_data = amount_data
        self.amount_seq = 0
        # data = pd.read_csv("./data/question_and_answer_no_code01/Dataset_Python_Question_Answer.csv", chunksize=10000)
        # data = pd.read_csv("data/WikiQACorpus/WikiQA-train.tsv", chunksize=10000,sep="\t")
        with open("/home/athip/psu/learning_AI/Text_Gen/data/PythonCodeDataSmall_TextOnly/Python_code_data.txt","r") as f:
            data = f.read(-1)
            data = data.split("\n# ")
            data = [data[0].strip("\n")] + [("# " + c).strip("\n") for c in data[1:] if len(c) >= 150]
        if os.path.isfile(self.qaaidx_path):
            if tokenizer_model == "bpe":
                self.tokenizer = BPE()
                self.tokenizer.load_pretrain(pretrain_model_tokenizer_path)
                with open(self.qaaidx_path,"rb") as f:
                    self.answer_all, self.answer_train_in, self.answer_train_out = pickle.load(f)
        else:
            # for d in tqdm(data):
                # self.question += d["Question"].tolist()
                # ans = d["Answer"].map(lambda x: "\n".join(eval(x))).tolist()
                # self.answer += ans
                # self.question += d["Question"].tolist()
                # self.answer += d["Sentence"].tolist()
                # self.answer += 
            self.answer = data
            del data
            if tokenizer_model == "bpe":
                self.tokenizer = BPE()
                self.tokenizer.load_pretrain(pretrain_model_tokenizer_path)
                for a in tqdm(self.answer):
                    answer_all = [1, ] + self.tokenizer.token2idx(self.tokenizer.tokenize(a)) + [2, ]
                    self.answer_all.append(answer_all)
                    for i in range(2,len(answer_all)-1):
                        self.answer_train_in.append(answer_all[0:i])
                        self.answer_train_out.append(answer_all[0:i+1])
                    # self.answer_train.append(self.tokenizer.token2idx(self.tokenizer.tokenize(a)) + [2, ])
                if (self.qaaidx_path != None) and (self.qaaidx_path[-4:] == ".pkl") and qaaidx_save:
                    with open(self.qaaidx_path,"wb") as f:
                        pickle.dump([self.answer_all, self.answer_train_in, self.answer_train_out],f)
        self.token_size = len(self.tokenizer.vocab)
        self.amount_seq = sum([len(i)-3 for i in self.answer_all[:self.amount_data]])


    def __len__(self):
        #return sum([len(data) for data in pd.read_csv(self.data_path,chunksize=10)])
        return self.amount_seq#len(self.answer_train_in)

    def __getitem__(self, index):
        # question_train = torch.tensor(self.question_train[index],device=self.device)
        # question_train = torch.nn.functional.pad(question_train,(0,self.window_size - len(question_train)),"constant",0)
        # answer_train = self.answer_train[index]
        # s_posi = random.randrange(0, len(answer_train) - 2, 1)
        # r_posi = random.randrange(s_posi + 1, len(answer_train) - 1, 1)
        # answer_train_in = torch.tensor(answer_train[s_posi:r_posi],device=self.device)
        # answer_train_out = torch.tensor(answer_train[s_posi:r_posi+1],device=self.device)
        answer_train_in = torch.tensor(self.answer_train_in[index],device=self.device)
        answer_train_out = torch.tensor(self.answer_train_out[index],device=self.device)
        answer_train_in = torch.nn.functional.pad(answer_train_in,(0,self.window_size - len(answer_train_in)),"constant",0)
        answer_train_out = torch.nn.functional.pad(answer_train_out,(0,self.window_size - len(answer_train_out)),"constant",0)
        # print(answer_train_in)
        # print(answer_train_out)
        return answer_train_in,answer_train_out

    def get_vocab(self):
        return self.tokenizer.vocab
    def get_weight(self):
        self.answer_feq = Counter([t for a in self.answer_train_out[:self.amount_seq] for t in a])
        all_value = sum([len(Tvalues) for Tvalues in self.answer_train_in])
        min_value = min(self.answer_feq.values())
        max_ratio = all_value/min_value
        sum_value = sum(self.answer_feq.values())
        self.weight = torch.tensor([((all_value/(self.answer_feq[t] + 1))/max_ratio)**(1/1.0) for t in range(len(self.tokenizer.vocab))]) #2.71828182846
        # print(self.weight.max())
        # time.sleep(100)
        # print(self.answer_feq)
        # print(self.weight)
        # print(self.weight.min())
        # print(all_value/(self.answer_feq[103] + 1))
        # print(all_value)
        # print(self.tokenizer.vocab)
        # time.sleep(100)
        return self.weight
    def get_sample(self):
        return ["".join(self.tokenizer.idx2token(tok)).replace("Ġ"," ").replace("Ċ","\n") for tok in self.answer_all[:self.amount_data]]
    

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, base_lr, start_step):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_max_steps = max_steps
        self.current_max_steps = max_steps
        self.base_lr = base_lr
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps), last_epoch=-1, eta_min=5e-6) #lr=5e-6
        self.current_step = 0
        if start_step != None:
            self.current_step = start_step
            self.cosine_scheduler.step(start_step)

    def step(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine annealing
            self.cosine_scheduler.step()
        # self.current_max_steps = (2**(self.current_step//self.start_max_steps))*self.start_max_steps
        # if (self.current_step%self.current_max_steps) and (self.current_step != 0) == 0:
        #     self.cosine_scheduler.T_max = self.current_max_steps
        
        self.current_step += 1
