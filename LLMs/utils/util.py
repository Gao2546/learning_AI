from collections import defaultdict, Counter
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
import pandas as pd
import torch
import random
import os
import sys
import time
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import Dataset as DatasetLoad

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer, Metaspace
from tokenizers.normalizers import Sequence as NormalizerSequence, Replace, NFKC, Lowercase # Added Replace, Sequence, etc.
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence, Metaspace, Split # Added Split, Sequence

from matplotlib import pyplot as plt


tokenizer = AutoTokenizer.from_pretrained("gpt2")


class BPEs:
    def __init__(self, vocab_size=5120):
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        self.normalizer = NormalizerSequence([
            # NFKC(), # Optional: Unicode normalization
            # Lowercase(), # Optional: Convert to lowercase
            Replace("\n", "Ċ") # Replace newline with Ċ
        ])
        self.tokenizer.normalizer = self.normalizer
        # self.tokenizer.pre_tokenizer = Metaspace(replacement="Ġ")
        self.pre_tokenizer = PreTokenizerSequence([
            Split(pattern="Ċ", behavior="isolated"), # Treat Ċ as a separate pre-token
            Metaspace(replacement="Ġ", prepend_scheme="never") # Handle spaces/prefixes for other parts
        ])
        self.tokenizer.pre_tokenizer = self.pre_tokenizer

        # Define trainer with small vocab size
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|pad|>", 
                            "<|startoftext|>", 
                            "<|unk|>", 
                            "<|endoftext|>",
                            "Ċ", # Add our custom newline token here
                            ],
            show_progress=True,
        )

    def train(self, files):
        # Train on a text corpus (plain text file paths)
        self.tokenizer.train(files, self.trainer)

        # Save the tokenizer
        self.tokenizer.save("./model/BPE_model/tokenizer-bpe-5k.json")

    def load(self, path):
        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(path)
        # self.tokenizer.pre_tokenizer = Metaspace(replacement="Ġ")
        self.tokenizer.normalizer = self.normalizer
        self.tokenizer.pre_tokenizer = self.pre_tokenizer
        # self.tokenizer.enable_truncation(5120)
        # self.tokenizer.enable_padding(pad_id=0, pad_token="<|pad|>", length=5120)

    def save(self, path):
        # Save the tokenizer
        self.tokenizer.save(path)

    def encode(self, text: str):
        """Encodes a piece of text."""
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]):
        """Decodes a list of token IDs back to text."""
        # Note: This will decode 'Ċ' as 'Ċ'. If you need '\n' back,
        # you'll need to replace it manually after decoding.
        # The 'Ġ' characters will likely remain as well.
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def decode_clean(self, ids: list[int]):
        """Decodes IDs and performs basic cleanup (Ċ -> \n, Ġ -> space)."""
        decoded_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        # Replace the custom newline token back to a standard newline
        # Replace the Metaspace prefix (often you want a space instead)
        # Use strip() to remove leading/trailing whitespace potentially introduced
        cleaned_text = decoded_text.replace(" ","").replace("Ċ", "\n").replace("Ġ", " ").strip()
        # Handle potential double spaces resulting from replacements
        # import re
        # cleaned_text = re.sub(r' +', ' ', cleaned_text)
        return cleaned_text

class BPEs2:
    def __init__(self, vocab_size=5120):
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        self.normalizer = NormalizerSequence([
            # NFKC(), # Optional: Unicode normalization
            # Lowercase(), # Optional: Convert to lowercase
            Replace("\n", "Ċ") # Replace newline with Ċ
        ])
        self.tokenizer.normalizer = self.normalizer
        # self.tokenizer.pre_tokenizer = Metaspace(replacement="Ġ")
        self.pre_tokenizer = PreTokenizerSequence([
            Split(pattern="Ċ", behavior="isolated"), # Treat Ċ as a separate pre-token
            Metaspace(replacement="Ġ", prepend_scheme="never") # Handle spaces/prefixes for other parts
        ])
        self.tokenizer.pre_tokenizer = self.pre_tokenizer

        # Define trainer with small vocab size
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|pad|>", 
                            "<|startoftext|>", 
                            "<|unk|>", 
                            "<|endoftext|>",
                            "Ċ", # Add our custom newline token here
                            ],
            show_progress=True,
        )

    def train(self, path, method = 2):
        # Train on a text corpus (plain text file paths)
        if method == 1:
            self.tokenizer.train(path, self.trainer)
        elif method == 2:
            if not os.path.isdir(path[0] + "train" if isinstance(path, list) else path + "train"):
                print("Directory does not exist.")
                # load_dataset(path="jtatman/python-code-dataset-500k", save_infos=True).save_to_disk(path[0] if isinstance(path, list) else path)
                load_dataset(path="papahawk/conversational-01", save_infos=True).save_to_disk(path[0] if isinstance(path, list) else path)
            data = load_dataset(path=path[0] if isinstance(path, list) else path, split="train")
            data = data.to_dict()
            data = data["response"] + data["prompt"]
            self.tokenizer.train_from_iterator(data)

        # Save the tokenizer
        # self.tokenizer.save(f"./model/BPE_model/tokenizer-bpe-{self.trainer.vocab_size // 1000}k.json")
        self.tokenizer.save(f"./model/BPE_model/tokenizer-bpe-conversational-{self.trainer.vocab_size // 1000}k.json")

    def load(self, path):
        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(path)
        # self.tokenizer.pre_tokenizer = Metaspace(replacement="Ġ")
        self.tokenizer.normalizer = self.normalizer
        self.tokenizer.pre_tokenizer = self.pre_tokenizer
        # self.tokenizer.enable_truncation(5120)
        # self.tokenizer.enable_padding(pad_id=0, pad_token="<|pad|>", length=5120)

    def save(self, path):
        # Save the tokenizer
        self.tokenizer.save(path)

    def encode(self, text: str):
        """Encodes a piece of text."""
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]):
        """Decodes a list of token IDs back to text."""
        # Note: This will decode 'Ċ' as 'Ċ'. If you need '\n' back,
        # you'll need to replace it manually after decoding.
        # The 'Ġ' characters will likely remain as well.
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def decode_clean(self, ids: list[int]):
        """Decodes IDs and performs basic cleanup (Ċ -> \n, Ġ -> space)."""
        decoded_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        # Replace the custom newline token back to a standard newline
        # Replace the Metaspace prefix (often you want a space instead)
        # Use strip() to remove leading/trailing whitespace potentially introduced
        cleaned_text = decoded_text.replace(" ","").replace("Ċ", "\n").replace("Ġ", " ").strip()
        # Handle potential double spaces resulting from replacements
        # import re
        # cleaned_text = re.sub(r' +', ' ', cleaned_text)
        return cleaned_text
    

class BPEsQA:
    def __init__(self, vocab_size=5120):
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        self.normalizer = NormalizerSequence([
            # NFKC(), # Optional: Unicode normalization
            # Lowercase(), # Optional: Convert to lowercase
            Replace("\n", "Ċ") # Replace newline with Ċ
        ])
        self.tokenizer.normalizer = self.normalizer
        # self.tokenizer.pre_tokenizer = Metaspace(replacement="Ġ")
        self.pre_tokenizer = PreTokenizerSequence([
            Split(pattern="Ċ", behavior="isolated"), # Treat Ċ as a separate pre-token
            Metaspace(replacement="Ġ", prepend_scheme="never") # Handle spaces/prefixes for other parts
        ])
        self.tokenizer.pre_tokenizer = self.pre_tokenizer

        # Define trainer with small vocab size
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|pad|>", 
                            "<|startoftext|>", 
                            "<|unk|>", 
                            "<|endoftext|>",
                            "Ċ", # Add our custom newline token here
                            "<|Q:|>",
                            "<|A:|>"
                            ],
            show_progress=True,
        )

    def train(self, path, method = 2):
        # Train on a text corpus (plain text file paths)
        if method == 1:
            self.tokenizer.train(path, self.trainer)
        elif method == 2:
            if not os.path.isdir(path[0] + "train" if isinstance(path, list) else path + "train"):
                print("Directory does not exist.")
                # load_dataset(path="jtatman/python-code-dataset-500k", save_infos=True).save_to_disk(path[0] if isinstance(path, list) else path)
                load_dataset(path="papahawk/conversational-01", save_infos=True).save_to_disk(path[0] if isinstance(path, list) else path)
            # data = load_dataset(path=path[0] if isinstance(path, list) else path, split="train")
            data = load_from_disk(dataset_path = path[0] if isinstance(path, list) else path)["train"]
            data = data.to_dict()
            print(data.keys())
            data = data["response"] + data["prompt"]
            self.tokenizer.train_from_iterator(data, trainer=self.trainer)

        # Save the tokenizer
        # self.tokenizer.save(f"./model/BPE_model/tokenizer-bpe-{self.trainer.vocab_size // 1000}k.json")
        self.tokenizer.save(f"./model/BPE_model/tokenizer-bpe-conversational-{self.trainer.vocab_size // 1000}k.json")

    def load(self, path):
        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(path)
        # self.tokenizer.pre_tokenizer = Metaspace(replacement="Ġ")
        self.tokenizer.normalizer = self.normalizer
        self.tokenizer.pre_tokenizer = self.pre_tokenizer
        # self.tokenizer.enable_truncation(5120)
        # self.tokenizer.enable_padding(pad_id=0, pad_token="<|pad|>", length=5120)

    def save(self, path):
        # Save the tokenizer
        self.tokenizer.save(path)

    def encode(self, text: str):
        """Encodes a piece of text."""
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]):
        """Decodes a list of token IDs back to text."""
        # Note: This will decode 'Ċ' as 'Ċ'. If you need '\n' back,
        # you'll need to replace it manually after decoding.
        # The 'Ġ' characters will likely remain as well.
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def decode_clean(self, ids: list[int]):
        """Decodes IDs and performs basic cleanup (Ċ -> \n, Ġ -> space)."""
        decoded_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        # Replace the custom newline token back to a standard newline
        # Replace the Metaspace prefix (often you want a space instead)
        # Use strip() to remove leading/trailing whitespace potentially introduced
        cleaned_text = decoded_text.replace(" ","").replace("Ċ", "\n").replace("Ġ", " ").replace("<|Q:|>", "Q: ").replace("<|A:|>","\nA: ").strip()
        # Handle potential double spaces resulting from replacements
        # import re
        # cleaned_text = re.sub(r' +', ' ', cleaned_text)
        return cleaned_text



# class BPE:
#     def __init__(self):
#         self.vocab = []
#         self.merges = {}
#         self.splits = {}
#         self.word_freqs = defaultdict(int)

#     def compute_pair_freqs(self):
#         pair_freqs = defaultdict(int)
#         for word, freq in self.word_freqs.items():
#             split = self.splits[word]
#             if len(split) == 1:
#                 continue
#             for i in range(len(split) - 1):
#                 pair = (split[i], split[i + 1])
#                 pair_freqs[pair] += freq
#         return pair_freqs

#     def merge_pair(self, a, b):
#         for word in self.word_freqs:
#             split = self.splits[word]
#             if len(split) == 1:
#                 continue

#             i = 0
#             while i < len(split) - 1:
#                 if split[i] == a and split[i + 1] == b:
#                     split = split[:i] + [a + b] + split[i + 2:]
#                     i += 1
#                 else:
#                     i += 1
#             self.splits[word] = split
#         return self.splits

#     def train(self, corpus, vocab_size):
#         print("count word freq")
#         for text in tqdm(corpus):
#             words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
#                 text)
#             new_words = [word for word, offset in words_with_offsets]
#             for word in new_words:
#                 self.word_freqs[word] += 1

#         alphabet = []
#         print("find alphabet")
#         for word in tqdm(self.word_freqs.keys()):
#             for letter in word:
#                 if letter not in alphabet:
#                     alphabet.append(letter)
#         alphabet.sort()
#         self.vocab = ["<|pad|>", "<|startoftext|>", "<|endoftext|>"] + alphabet.copy()
#         self.splits = {word: [c for c in word]
#                        for word in self.word_freqs.keys()}
#         with tqdm(total=vocab_size) as pbar:
#             pbar.update(len(self.vocab))
#             while (len(self.vocab) < vocab_size):
#                 pair_freqs = self.compute_pair_freqs()
#                 if len(pair_freqs) == 0:
#                     break
#                 best_pair = ""
#                 max_freq = None
#                 # best_pair = max(pair_freqs, key=pair_freqs.get)
#                 # max_freq = pair_freqs[best_pair]
#                 for pair, freq in pair_freqs.items():
#                     if max_freq is None or max_freq < freq:
#                         best_pair = pair
#                         max_freq = freq
#                 self.splits = self.merge_pair(
#                     *best_pair)
#                 self.merges[best_pair] = best_pair[0] + best_pair[1]
#                 self.vocab.append(best_pair[0] + best_pair[1])
#                 pbar.update(1)

#     def tokenize(self, text):
#         pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(
#             text)
#         pre_tokenized_text = [word for word, offset in pre_tokenize_result]
#         splits = [[l for l in word] for word in pre_tokenized_text]
#         for pair, merge in self.merges.items():
#             for idx, split in enumerate(splits):
#                 i = 0
#                 while i < len(split) - 1:
#                     if split[i] == pair[0] and split[i + 1] == pair[1]:
#                         split = split[:i] + [merge] + split[i + 2:]
#                     else:
#                         i += 1
#                 splits[idx] = split

#         return sum(splits, [])
    
#     def token2idx(self, token):
#         return [self.vocab.index(t) for t in token]
    
#     def idx2token(self, idx):
#         return [self.vocab[ids] for ids in idx]

#     def decode(self, tokens):
#         sentence = "".join(tokens).replace("Ġ", " ")
#         return sentence

#     def load_pretrain(self, path):
#         with open(path, "rb") as f:
#             data = pickle.load(f)
#         self.vocab = data["vocab"]
#         self.merges = data["merges"]
#         self.splits = data["splits"]
#         self.word_freqs = data["word_freqs"]

#     def save_model(self, path):
#         data = {"vocab": self.vocab,
#                 "merges": self.merges,
#                 "splits": self.splits,
#                 "word_freqs": self.word_freqs}
#         with open(path, 'wb') as f:
#             pickle.dump(data, f)

#     def add_vocabs(self, new_vocabs: list):
#         self.vocab = new_vocabs + self.vocab


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
        with open("./data/PythonCodeDataSmall_TextOnly/Python_code_data.txt","r") as f:
            data = f.read(-1)
            data = data.split("\n# ")
            data = [data[0].strip("\n")] + [("# " + c).strip("\n") for c in data[1:] if len(c) >= 80]
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
                    for i in range(2,len(answer_all)):
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
    

class data_loader(Dataset):
    def __init__(self, path, new_tokenizer, max_len=512):
        self.max_len = max_len
        self.new_tokenizer = new_tokenizer
        self.data_path = path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(self.data_path, "r") as f:
            data = f.read(-1)
            data = data.split("\n# ")
            data = [data[0].strip("\n")] + [("# " + c).strip("\n") for c in data[1:] if len(c) >= 80]
        self.pre_data = data
        print(len(self.pre_data))
        # self.tokens_data_new = new_tokenizer.tokenize(data)
        # tt = [F.pad(torch.tensor(new_tokenizer.tokenizer.encode(dd).ids, dtype=torch.int), mode='constant', pad=(0, max(512 - len(new_tokenizer.tokenizer.encode(dd).tokens), 0)), value=0) for dd in self.pre_data]
        # self.tokens_data_new = torch.stack(tt)
    def __len__(self):
        return len(self.pre_data)

    def __getitem__(self, idx):
        data_token = torch.tensor([1] + self.new_tokenizer.tokenizer.encode(self.pre_data[idx]).ids + [3], device=self.device)
        data_token = data_token[0:random.randint(10, data_token.shape[0])]
        data_token_in = data_token[:-1].clone()
        data_token_out = data_token[:].clone()
        data_token_in_pad = F.pad(data_token_in, mode='constant', pad=(0, max(self.max_len - len(data_token_in), -1000000)), value=0)
        data_token_out_pad = F.pad(data_token_out, mode='constant', pad=(0, max(self.max_len - len(data_token_out), -1000000)), value=0)
        return data_token_in_pad, data_token_out_pad
    def get_sample(self):
        rr = random.randint(0, len(self.pre_data)-1)
        return self.pre_data[rr:rr+10]
    def get_vocab(self):
        return self.new_tokenizer.vocab
    

class data_loader2(Dataset):
    def __init__(self, path, new_tokenizer, max_len=512):
        self.max_len = max_len
        self.new_tokenizer = new_tokenizer
        self.data_path = path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(self.data_path, "r") as f:
            data = f.read(-1)
            data = data.split("\n\n# ")
            data = [[1] + new_tokenizer.tokenizer.encode(dd).ids + [3] for dd in data]
            data = [[new_tokenizer.decode_clean(dds[:r]) for r in range(2,min(len(dds),max_len),1 )] \
                  + [new_tokenizer.decode_clean(dds[j:j + max_len]) for j in range(1,len(dds) - max_len, 1)] \
                            for dds in data]
            flattened_list = list(itertools.chain.from_iterable(data))
            self.pre_data = flattened_list
            # data = [data[0].strip("\n")] + [("# " + c).strip("\n") for c in data[1:] if len(c) >= 80]
        print(len(self.pre_data))
        # self.tokens_data_new = new_tokenizer.tokenize(data)
        # tt = [F.pad(torch.tensor(new_tokenizer.tokenizer.encode(dd).ids, dtype=torch.int), mode='constant', pad=(0, max(512 - len(new_tokenizer.tokenizer.encode(dd).tokens), 0)), value=0) for dd in self.pre_data]
        # self.tokens_data_new = torch.stack(tt)
    def __len__(self):
        return len(self.pre_data)

    def __getitem__(self, idx):
        # data_token = torch.tensor([1] + self.new_tokenizer.tokenizer.encode(self.pre_data[idx]).ids + [3], device=self.device)
        data_token = torch.tensor(self.new_tokenizer.tokenizer.encode(self.pre_data[idx]).ids, device=self.device, dtype=torch.long)
        # data_token = data_token[0:random.randint(10, data_token.shape[0])]
        data_token_in = data_token[:-1].clone()
        data_token_out = data_token[:].clone()
        data_token_in_pad = F.pad(data_token_in, mode='constant', pad=(0, max(self.max_len - len(data_token_in), -1000000)), value=0)
        data_token_out_pad = F.pad(data_token_out, mode='constant', pad=(0, max(self.max_len - len(data_token_out), -1000000)), value=0)
        return data_token_in_pad, data_token_out_pad
    def get_sample(self):
        rr = random.randint(0, len(self.pre_data)-1)
        return self.pre_data[rr:rr+10]
    def get_vocab(self):
        return self.new_tokenizer.vocab
    
class data_loader3(Dataset):
    def __init__(self, path, new_tokenizer, max_len=512):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_len = max_len
        self.new_tokenizer = new_tokenizer
        self.data_path = path
        if not os.path.isdir(self.data_path+"train"):
            print("Directory does not exist.")
            # load_dataset(path="jtatman/python-code-dataset-500k", save_infos=True).save_to_disk(self.data_path)
            load_dataset(path="papahawk/conversational-01", save_infos=True).save_to_disk(self.data_path)
        data = load_dataset(path=self.data_path, split="train")
        self.pre_data = data
        print(len(self.pre_data))
        # self.tokens_data_new = new_tokenizer.tokenize(data)
        # tt = [F.pad(torch.tensor(new_tokenizer.tokenizer.encode(dd).ids, dtype=torch.int), mode='constant', pad=(0, max(512 - len(new_tokenizer.tokenizer.encode(dd).tokens), 0)), value=0) for dd in self.pre_data]
        # self.tokens_data_new = torch.stack(tt)
    def __len__(self):
        return 4#int(len(self.pre_data)*0.01)
    def __getitem__(self, idx):
        # print(self.pre_data[idx]["instruction"])
        question = torch.tensor(self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["instruction"]).ids, device=self.device)
        answer = torch.tensor([1] + self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["output"]).ids + [3], device=self.device)
        # print(answer.size())
        # rr = random.randint(0, len(answer)-self.max_len if len(answer) - self.max_len > 0 else 2)
        rr = random.randint(0, min(len(answer), self.max_len))
        # answer = answer[0:random.randint(1, answer.shape[0])]
        answer = answer[0:rr]
        answer_in = answer[:-1].clone()
        answer_out = answer.clone()

        question_pad = F.pad(question, mode='constant', pad=(0, max(self.max_len - len(question), -1000000)), value=0)
        answer_in_pad = F.pad(answer_in, mode='constant', pad=(0, max(self.max_len - len(answer_in), -1000000)), value=0)
        answer_out_pad = F.pad(answer_out, mode='constant', pad=(0, max(self.max_len - len(answer_out), -1000000)), value=0)

        # data_token = torch.tensor([1] + self.new_tokenizer.tokenizer.encode(self.pre_data[idx]).ids + [3], device=self.device)
        # data_token = data_token[0:random.randint(10, data_token.shape[0])]
        # data_token_in = data_token[:-1].clone()
        # data_token_out = data_token[:].clone()
        # data_token_in_pad = F.pad(data_token_in, mode='constant', pad=(0, max(self.max_len - len(data_token_in), -1000000)), value=0)
        # data_token_out_pad = F.pad(data_token_out, mode='constant', pad=(0, max(self.max_len - len(data_token_out), -1000000)), value=0)
        return question_pad, answer_in_pad, answer_out_pad
    def get_sample(self):
        rr = random.randint(0, len(self.pre_data)-1)
        rr = 0
        return self.pre_data.to_dict()['output'][rr:rr+3] + self.pre_data.to_dict()['instruction'][rr:rr+3]
    def get_vocab(self):
        return self.new_tokenizer.vocab
    
        
class data_loaderQA(Dataset):
    def __init__(self, path, new_tokenizer, max_len=512, data_path512=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_len = max_len
        self.new_tokenizer = new_tokenizer
        self.data_path = path
        self.data_path512 = data_path512
        if not os.path.isdir(self.data_path+"train"):
            print("Directory does not exist.")
            # load_dataset(path="jtatman/python-code-dataset-500k", save_infos=True).save_to_disk(self.data_path)
            load_dataset(path="papahawk/conversational-01", save_infos=True).save_to_disk(self.data_path)
        # data = load_dataset(path=self.data_path, split="train")
        data = load_from_disk(dataset_path = self.data_path)["train"]
        # self.pre_data = data
        def is_valid(example):
            question = self.new_tokenizer.tokenizer.encode(example["prompt"]).ids
            answer = self.new_tokenizer.tokenizer.encode(example["response"]).ids
            QA_data = [1] + [5] + question + [6] + answer + [3]
            return len(QA_data) <= self.max_len
        if len(os.listdir(self.data_path512)) <= 1:
            self.pre_data = data.filter(is_valid)
            self.pre_data.save_to_disk(self.data_path512)
        else:
            data = None
            self.pre_data = load_from_disk(self.data_path512)
        print(f"Filtered dataset size: {len(self.pre_data)}")
        # print(len(self.pre_data))
        
        # self.tokens_data_new = new_tokenizer.tokenize(data)
        # tt = [F.pad(torch.tensor(new_tokenizer.tokenizer.encode(dd).ids, dtype=torch.int), mode='constant', pad=(0, max(512 - len(new_tokenizer.tokenizer.encode(dd).tokens), 0)), value=0) for dd in self.pre_data]
        # self.tokens_data_new = torch.stack(tt)
    def __len__(self):
        return 8#int(len(self.pre_data)*0.001)
    def __getitem__(self, idx):
        # print(self.pre_data[idx]["instruction"])
        # question = torch.tensor(self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["instruction"]).ids, device=self.device)
        # answer = torch.tensor([1] + self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["output"]).ids + [3], device=self.device)

        question = self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["prompt"]).ids
        answer = self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["response"]).ids

        QA_data = torch.tensor([1] + [5] + question + [6] + answer + [3], device=self.device)

        # print(answer.size())
        # rr = random.randint(0, len(answer)-self.max_len if len(answer) - self.max_len > 0 else 2)
        rr = random.randint(len(question) + 3, min(len(QA_data), self.max_len))
        # answer = answer[0:random.randint(1, answer.shape[0])]
        QA_data = QA_data[0:rr]
        QA_in = QA_data[:-1].clone()
        QA_out = QA_data.clone()

        # question_pad = F.pad(question, mode='constant', pad=(0, max(self.max_len - len(question), -1000000)), value=0)
        QA_in_pad = F.pad(QA_in, mode='constant', pad=(0, max(self.max_len - len(QA_in), -1000000)), value=0)
        QA_out_pad = F.pad(QA_out, mode='constant', pad=(0, max(self.max_len - len(QA_out), -1000000)), value=0)

        # data_token = torch.tensor([1] + self.new_tokenizer.tokenizer.encode(self.pre_data[idx]).ids + [3], device=self.device)
        # data_token = data_token[0:random.randint(10, data_token.shape[0])]
        # data_token_in = data_token[:-1].clone()
        # data_token_out = data_token[:].clone()
        # data_token_in_pad = F.pad(data_token_in, mode='constant', pad=(0, max(self.max_len - len(data_token_in), -1000000)), value=0)
        # data_token_out_pad = F.pad(data_token_out, mode='constant', pad=(0, max(self.max_len - len(data_token_out), -1000000)), value=0)
        return QA_in_pad, QA_out_pad
    def get_sample(self):
        rr = random.randint(0, len(self.pre_data)-1)
        rr = 0
        return self.pre_data.to_dict()['response'][rr:1] , self.pre_data.to_dict()['prompt'][rr:1]
    def get_vocab(self):
        return self.new_tokenizer.vocab
    

class data_loaderQA(Dataset):
    def __init__(self, path, new_tokenizer, max_len=512, data_path512=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_len = max_len
        self.new_tokenizer = new_tokenizer
        self.data_path = path
        self.data_path512 = data_path512
        if not os.path.isdir(self.data_path+"train"):
            print("Directory does not exist.")
            # load_dataset(path="jtatman/python-code-dataset-500k", save_infos=True).save_to_disk(self.data_path)
            load_dataset(path="papahawk/conversational-01", save_infos=True).save_to_disk(self.data_path)
        # data = load_dataset(path=self.data_path, split="train")
        data = load_from_disk(dataset_path = self.data_path)["train"]
        # self.pre_data = data
        def is_valid(example):
            question = self.new_tokenizer.tokenizer.encode(example["prompt"]).ids
            answer = self.new_tokenizer.tokenizer.encode(example["response"]).ids
            QA_data = [1] + [5] + question + [6] + answer + [3]
            return len(QA_data) <= self.max_len
        if len(os.listdir(self.data_path512)) <= 1:
            self.pre_data = data.filter(is_valid,num_proc=8)
            self.pre_data.save_to_disk(self.data_path512)
        else:
            data = None
            self.pre_data = load_from_disk(self.data_path512)
        print(f"Filtered dataset size: {len(self.pre_data)}")
        # print(len(self.pre_data))
        
        # self.tokens_data_new = new_tokenizer.tokenize(data)
        # tt = [F.pad(torch.tensor(new_tokenizer.tokenizer.encode(dd).ids, dtype=torch.int), mode='constant', pad=(0, max(512 - len(new_tokenizer.tokenizer.encode(dd).tokens), 0)), value=0) for dd in self.pre_data]
        # self.tokens_data_new = torch.stack(tt)
    def __len__(self):
        return int(len(self.pre_data)*0.01)
    def __getitem__(self, idx:int):
        # print(self.pre_data[idx]["instruction"])
        # question = torch.tensor(self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["instruction"]).ids, device=self.device)
        # answer = torch.tensor([1] + self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["output"]).ids + [3], device=self.device)

        question = self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["prompt"]).ids
        answer = self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["response"]).ids

        lenght_answer = torch.tensor([len(answer)], device=self.device)

        QA_data = torch.tensor([1] + [5] + question + [6] + answer + [3], device=self.device)

        # print(answer.size())
        # rr = random.randint(0, len(answer)-self.max_len if len(answer) - self.max_len > 0 else 2)
        rr = random.randint(len(question) + 4, min(len(QA_data)+1, self.max_len))
        # answer = answer[0:random.randint(1, answer.shape[0])]
        QA_data = QA_data[0:rr]
        QA_in = QA_data[:-1].clone()
        QA_out = QA_data.clone()

        # question_pad = F.pad(question, mode='constant', pad=(0, max(self.max_len - len(question), -1000000)), value=0)
        QA_in_pad = F.pad(QA_in, mode='constant', pad=(0, max(self.max_len - len(QA_in), -1000000)), value=0)
        QA_out_pad = F.pad(QA_out, mode='constant', pad=(0, max(self.max_len - len(QA_out), -1000000)), value=0)

        # data_token = torch.tensor([1] + self.new_tokenizer.tokenizer.encode(self.pre_data[idx]).ids + [3], device=self.device)
        # data_token = data_token[0:random.randint(10, data_token.shape[0])]
        # data_token_in = data_token[:-1].clone()
        # data_token_out = data_token[:].clone()
        # data_token_in_pad = F.pad(data_token_in, mode='constant', pad=(0, max(self.max_len - len(data_token_in), -1000000)), value=0)
        # data_token_out_pad = F.pad(data_token_out, mode='constant', pad=(0, max(self.max_len - len(data_token_out), -1000000)), value=0)
        return QA_in_pad, QA_out_pad, lenght_answer
    def get_sample(self):
        rr = random.randint(0, len(self.pre_data)-1)
        rr = 0
        return self.pre_data.to_dict()['response'][rr:1] , self.pre_data.to_dict()['prompt'][rr:1]
    def get_vocab(self):
        return self.new_tokenizer.vocab
    

class data_loaderQA_SEQ(Dataset):
    def __init__(self, path, new_tokenizer, max_len=512, data_path512=None, data_path512_seq=None, data_sector = 0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_len = max_len
        self.new_tokenizer = new_tokenizer
        self.data_path = path
        self.data_path512 = data_path512
        self.data_path512_seq = data_path512_seq
        self.data_sector = data_sector
        if not os.path.isdir(self.data_path+"train"):
            print("Directory does not exist.")
            load_dataset(path="papahawk/conversational-01", save_infos=True).save_to_disk(self.data_path)
        data = load_from_disk(dataset_path = self.data_path)["train"]
        def is_valid(example):
            question = self.new_tokenizer.tokenizer.encode(example["prompt"]).ids
            answer = self.new_tokenizer.tokenizer.encode(example["response"]).ids
            QA_data = [1] + [5] + question + [6] + answer + [3]
            return len(QA_data) <= self.max_len
        
        def gen_seq():
            for data in tqdm(self.pre_data):
                question = self.new_tokenizer.tokenizer.encode(data["prompt"]).ids
                answer = self.new_tokenizer.tokenizer.encode(data["response"]).ids
                QA_data = [1] + [5] + question + [6] + answer + [3]
                for i in range(len(question)+4,len(QA_data) + 1):
                    yield {'prompt':QA_data[:i-1], 'response':QA_data[0:i]}
        if (len(os.listdir(self.data_path512)) <= 1) and (len(os.listdir(self.data_path512_seq)) <= 1):
            self.pre_data = data.filter(is_valid,num_proc=8)
            self.pre_data.save_to_disk(self.data_path512)
        if len(os.listdir(self.data_path512)) > 1 and len(os.listdir(self.data_path512_seq)) <= 1:
            data = None
            self.pre_data = load_from_disk(self.data_path512)
            self.pre_data = DatasetLoad.from_generator(gen_seq,num_proc=8)
            self.pre_data.save_to_disk(self.data_path512_seq)
        if len(os.listdir(self.data_path512)) > 1 and len(os.listdir(self.data_path512_seq)) > 1:
            data = None
            self.pre_data = load_from_disk(self.data_path512_seq)
        print(f"Filtered dataset size: {len(self.pre_data)}")

        # if len(os.listdir(self.data_path512)) <= 1:

        # data.set_format(type="torch", columns=["prompt", "response"])
        # q_list = []
        # a_list = []
        # for idx in range(len(self.pre_data)):
        #     question = self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["prompt"]).ids
        #     answer = self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["response"]).ids
        #     QA_data = [1] + [5] + question + [6] + answer + [3]
        #     for i in range(len(question)+3,len(QA_data)):
        #         q_list.append(QA_data[:i-1])
        #         a_list.append(QA_data[i:i])

    def __len__(self):
        return int(len(self.pre_data)*0.001) #int(len(self.pre_data)*0.1)
    def __getitem__(self, idx):

        # question = self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["prompt"]).ids
        # answer = self.new_tokenizer.tokenizer.encode(self.pre_data[idx]["response"]).ids

        # lenght_answer = torch.tensor([len(answer)], device=self.device)

        # QA_data = torch.tensor([1] + [5] + question + [6] + answer + [3], device=self.device)

        # rr = random.randint(len(question) + 3, min(len(QA_data), self.max_len))
        # QA_data = QA_data[0:rr]
        # QA_in = QA_data[:-1].clone()
        # QA_out = QA_data.clone()

        # test = torch.tensor([i for i in range(30)], device=self.device)
        # data_in = [int(i) for i in self.pre_data[idx]["prompt"][0]]
        # data_out = [int(i) for i in self.pre_data[idx]["response"][0]]
        # [print(i) for i in self.pre_data[0]["prompt"]]
        # print("++++++++++++++++++++++++++")
        # print(idx)
        idx = idx + int(len(self.pre_data)*0.01)*self.data_sector
        QA_in = torch.tensor(self.pre_data[idx]["prompt"], device=self.device)
        QA_out = torch.tensor(self.pre_data[idx]["response"], device=self.device)

        QA_in_pad = F.pad(QA_in, mode='constant', pad=(0, max(self.max_len - len(QA_in), -1000000)), value=0)
        QA_out_pad = F.pad(QA_out, mode='constant', pad=(0, max(self.max_len - len(QA_out), -1000000)), value=0)

        return QA_in_pad, QA_out_pad
    def get_sample(self):
        rr = random.randint(0, len(self.pre_data)-1)
        rr = 0
        # return self.pre_data.to_dict()['response'][rr:1] , self.pre_data.to_dict()['prompt'][rr:1]
        return self.pre_data[rr:rr+10]
    
    def get_vocab(self):
        return self.new_tokenizer.vocab
    

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, base_lr, start_step):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_max_steps = max_steps
        self.current_max_steps = max_steps
        self.base_lr = base_lr
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, last_epoch=-1, eta_min=1e-6) #lr=5e-6 1e-6 5e-7
        # self.cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max_steps, T_mult=1, last_epoch=-1, eta_min=5e-6) 
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


def check_and_create_folder(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)   

class check_loss:
    def __init__(self):
        self.his_loss = []
        self.his_epochs = []
        self.curr_loss = 0
        self.curr_epoch = 0
    def add(self,loss,epoch):
        self.curr_loss = loss
        self.curr_epoch = epoch
        self.his_loss.append(self.curr_loss)
        self.his_epochs.append(self.curr_epoch)
    def plot_save(self,file_path,para):
        fig = plt.figure(0)
        plt.plot(self.his_epochs, self.his_loss)
        fig.savefig(fname=f"{os.path.join(file_path,para)}.png")