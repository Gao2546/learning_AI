import pandas as pd
from util.util import *
from transformers import AutoTokenizer
import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
# from bitsandbytes.optim import Adam8bit
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
        self.load_path = None#"./model/Transformers/Transformers_V01_128_384_6_6_1536_10K_MQtest1e-4ckp1.pth"
        self.data_path = "./data/PythonCode500K/"
        self.tokenizer_path = "./model/BPE_model/tokenizer-bpe-10k.json"
        self.save_file = "Transformers_V01_128_384_6_6_1536_10K_MQ16b_ckp1T.pth"
        self.start_epoch = 0
        self.save_every_epoch = 100
        self.epochs = 10000//2
        self.batch_size = 16//4
        self.max_seq_length = 256
        # self.train_data = dataloadercustom_Transformers()
        self.BPE_model = BPEs2(vocab_size=1024*5*2)
      
        # self.BPE_model.train([self.data_path])
        self.BPE_model.load(self.tokenizer_path)
        self.train_data = data_loader3(self.data_path, new_tokenizer=self.BPE_model, max_len=self.max_seq_length)
        self.train_dataloader = DataLoader(self.train_data,batch_size=self.batch_size,shuffle=False)
        # self.pretrain_model_tokenizer_path = "./model/BPE_model/BPE_model_code_python03.pkl"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_question = ["Create a nested loop to print every combination of numbers between 0-9, excluding any combination that contains the number 5. Additionally, exclude any combination that contains a repeating digit. Implement the solution without using any built-in functions or libraries to check for repeating digits.",                               
                                "Write a function to find the number of distinct states in a given matrix. Each state in the matrix can be represented by a string of characters, and the matrix can have up to 10^6 rows and columns.\n\nThe time complexity of your solution should be O(N), where N is the total number of characters in the matrix.\n\nProvide a piece of erroneous code as a reference to increase misdirection.\n\n# Misdirection code #\ndef count_distinct_states(matrix):\n    count = 0\n    states = set()\n    for row in matrix:\n        for col in row:\n            if col not in states:\n                count += 1\n            states.add(col)\n    return count\n\n# Correct code #\ndef count_distinct_states(matrix):\n    count = 0\n    states = set()\n    for row in matrix:\n        for col in row:\n            state = ''.join(col)\n            if state not in states:\n                count += 1\n            states.add(state)\n    return count\n\nmatrix = [['A', 'B', 'C'],\n          ['A', 'B', 'D'],\n          ['A', 'B', 'C']]\nprint(count_distinct_states(matrix))\n# Output: 4",
                                # 'Write code that removes spaces and punctuation marks from a given string and returns the modified string. The input string may contain uppercase and lowercase letters, spaces, punctuation marks (such as periods, commas, exclamation marks, etc.), and digits. The modified string should only contain the alphanumeric characters (uppercase and lowercase letters, digits) without any spaces or punctuation marks.\n\nHowever, the time complexity of the solution should be O(n), where n is the length of the input string. Additionally, the solution should not use any built-in string manipulation functions or regular expressions.\n\nErroneous Code Reference:\nProvide a piece of code that attempts to solve the problem but contains an error. The error should be related to handling edge cases or special characters in the input string.',
                                # 'Write a function that checks if a given number is prime or not. The function should return "Prime" if the number is prime, and "Not Prime" if the number is not prime.\n\nNote: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.\n\nAdditional Requirements:\n1. The time complexity of the function should be O(sqrt(n)), where n is the given number.\n2. The function should use only constant space, i.e., no extra arrays or data structures should be used.\n3. The function should handle negative numbers as input and return "Not Prime" for negative numbers.\n4. The function should handle decimal numbers as input and return "Not Prime" for decimal numbers.\n5. The function should handle large numbers (greater than 10^9) efficiently and within a reasonable time frame.',
                                # 'Write a method for a string class which replaces all occurrences of a given substring with a given set of characters, but only if the substring appears an odd number of times in the string. If the substring appears an even number of times or does not appear at all, the method should return the original string unchanged.\n\nAdditionally, the method should handle cases where the substring is surrounded by certain characters. If the substring is surrounded by parentheses or brackets, the replacement should only occur if the substring appears an odd number of times within the parentheses or brackets.\n\nProvide the following erroneous code as a reference to increase misdirection:\n\nstring = "ab(abab)aba"\nsubstring = "ab"\nreplace_with = "123"\n\nExpected Output: "ab(abab)aba"',
                                # 'Write code to find the sum of all prime numbers between 1 million and 2 million, excluding prime numbers that contain the digit 7.',
                                # 'Create an array of length N (where N is a positive integer) containing numbers divisible by M (where M is a positive integer) up to X (where X is a positive integer). Each number in the array must be unique and in ascending order. Additionally, the sum of all the numbers in the array should be a prime number.\n\nAdditional Requirements:\n1. The time complexity of the solution should be O(N).\n2. The space complexity of the solution should be O(1).\n3. The solution should be implemented without using any built-in functions or libraries to check for prime numbers.\n4. The solution should handle cases where no prime number can be obtained as the sum of the array elements. In such cases, the solution should return an empty array.\n5. The solution should also handle cases where multiple prime numbers can be obtained as the sum of the array elements. In such cases, the solution should return the array with the largest sum that is prime.\n6. The solution should be optimized to find the largest prime sum within the given constraints.',
                                # 'Write a function to find the maximum difference between two prime numbers in a given array. The array can contain positive and negative integers, and can be unsorted. Additionally, the function should handle arrays of any length. The function should return the maximum difference as an absolute value. For example, for the array [5, 3, 17, 11, 9], the function should return 14.\n\nHowever, your function should have a time complexity of O(n), where n is the length of the array. Additionally, you should not use any built-in functions or libraries to check if a number is prime. You need to implement your own prime checking function.',
                                # 'Write a program that calculates the height of a triangle given the angle, side lengths, opposite side length, and the ratio of the side lengths. The program should take into account the Law of Sines and the Law of Cosines. Additionally, the program should simulate the effect of air resistance on the trajectory of the triangle when it is thrown at a certain velocity.',
                                # 'Create a function to calculate the area of a given circle. The input parameter for the radius should be a string instead of a number. Additionally, the function should handle invalid inputs and return an error message if the input is not a valid number.\n\nThe function should also validate that the radius is a positive number. If the radius is negative or zero, the function should return an error message.\n\nLastly, the function should return the calculated area as a string with exactly two decimal places.',
                                # 'Write a function to generate the nth Fibonacci number. The function should have a time complexity of O(log n) and use dynamic programming. Additionally, the function should only use a constant amount of space, without using any arrays or additional data structures.',
                                # 'Use the function to debug the given program and prevent the segmentation fault. Your solution should also handle the case where the array contains duplicate elements. You are not allowed to use any additional data structures. Additionally, the time complexity of your solution should be O(n) and the space complexity should be O(1).\n\n```python\ndef debug_program(arr):\n    n = len(arr)\n    for i in range(n):\n        if arr[i] == i:\n            return i\n    return -1\n\n# Test Case\narr = [0, 1, 2, 3, 4]\nprint(debug_program(arr))  # Expected output: -1\n```\n\n**Additional Requirements:**\n\n- The program should be able to handle arrays of any length.\n- The program should be able to handle arrays with duplicate elements.\n- The solution should use a divide and conquer approach to solve the problem.\n- The solution should not modify the input array.\n- The solution should be implemented in Python.',
                                # 'Modify the code to perform the mathematical expression "x to the power of y" while also ensuring that the value of x is between 1 and 10, and the value of y is between 0 and 5. Additionally, the code should handle any potential errors or edge cases that may arise during the calculation. The code should also check for invalid inputs such as non-numeric values for x and y, and display appropriate error messages. \n\nHowever, the program should now be able to handle extremely large numbers efficiently and accurately, even when x is a decimal number. The program should use a custom algorithm to calculate the result, rather than relying on any built-in math functions. The program should also provide the result in scientific notation if it exceeds a certain threshold, which should be dynamically determined based on the input values.',
                                # 'Sort the array in descending order without using any built-in sorting functions or libraries. The array may contain duplicate elements.\n\nConstraints:\n- The input array has a length of at most 10^6.\n- The elements in the array are integers between -10^9 and 10^9.\n\narr = [3, 2, 1, 5, 4]',
                                # 'Create a function that takes an array of integers as an argument and returns the sum of all the prime numbers in the array. If the array does not contain any prime numbers, return 0.\n\nExample:\n\nInput: [1, 2, 3, 4, 5]\nOutput: 10\n\nInput: [1, 3, 5]\nOutput: 9\n\nInput: [2, 4, 6, 8]\nOutput: 2\n\nInput: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nOutput: 17\n\nInput: []\nOutput: 0',
                                # 'Write a function that calculates the factorial of each number in the list using recursion, but without using any built-in libraries or mathematical operators.\narr = [1, 2, 3, 4, 5]',
                                # 'Count the number of vowels in the string and return a dictionary where the keys are the vowels and the values are the counts of each vowel. However, you are not allowed to use any built-in functions or libraries that directly count the number of vowels in the string.\n\ntext = "Hello World!"\n\nFor example, for the given text, the output should be:\n{\'a\': 0, \'e\': 1, \'i\': 0, \'o\': 2, \'u\': 0}',
                                # 'Generate a random number between 0 and 10 (inclusively) using only bitwise operations. The generated number should have exactly 4 bits set to 1. Additionally, provide a piece of erroneous code as a reference to increase misdirection.',
                                # 'Create a function that takes in two numbers as arguments and returns the product of the two. However, you are not allowed to use the multiplication operator or any built-in functions or methods that directly calculate the product of two numbers. Additionally, your solution should have a time complexity of O(log n), where n is the larger of the two input numbers. You should instead implement your own logic to calculate the product using only bitwise operations such as bitwise shift and bitwise AND, as well as basic arithmetic operations such as addition, subtraction, and division.',
                                # 'Design a program that finds the longest element in a given list, which should include at least 100 elements. The program should also handle negative numbers, floating-point numbers, and alphanumeric characters as elements. Additionally, it should return the longest element, its index in the list, and the total number of occurrences of the longest element in the list. The program should also ensure that it runs efficiently and has a time complexity of O(n).',
                                # 'Write an algorithm in Python to determine if a number is prime or composite. Your algorithm should have a time complexity of O(n^2).\n\nNote: You are not allowed to use any built-in functions or libraries to check if a number is prime. You have to implement the algorithm from scratch.\n\nExamples:\n1. Input: 2\n   Output: Prime\n\n2. Input: 9\n   Output: Composite',
                                # 'Write a function to print all prime numbers between two given numbers, excluding any prime numbers that contain the digit 5. Additionally, ensure that the function is optimized to handle larger inputs efficiently. The time complexity of the solution should be O(n log log n) where n is the difference between the two given numbers.',
                                # 'Create a list comprehension to generate a list of all prime numbers from 1000 to 2000, but with the following additional requirements:\n\n1. Implement a separate function to check if a number is prime. This function should take in a parameter and return a boolean value indicating whether the number is prime or not.\n\n2. Instead of generating the prime numbers from 1000 to 2000, generate them from 10000 to 20000.\n\n3. Implement a caching mechanism to store the previously calculated prime numbers so that the prime checking function can take advantage of it and avoid unnecessary calculations.\n\n4. Add a timer to measure the time it takes to generate the list of prime numbers.\n\n5. Modify the list comprehension to only include prime numbers that are palindromic, i.e., numbers that remain the same when their digits are reversed.\n\n6. Add an additional requirement that the list comprehension should only include prime numbers that are also Fibonacci numbers.',
                                # "Write a HTML code that creates a form for users to fill out their current address. The form should include the following fields: first name, last name, email address, phone number, street address, city, state, and zip code. Additionally, the form should include validation to ensure that all fields are filled out correctly before the form can be submitted. The validation should check that the email address is in the correct format, the phone number is in the correct format, and the zip code is a valid format for the given state. Furthermore, the form should also include a password field with validation to ensure that the password meets the following criteria: it must be at least 12 characters long, contain at least two uppercase letters, two lowercase letters, two numbers, and two special characters. Additionally, the form should include a dropdown menu for selecting the user's preferred programming language from a predefined list of options.",
                                # "Compose a function named average_list which takes a list of numbers as its argument and computes the average. The list may contain both positive and negative numbers.\n\nThe function should handle the following edge cases:\n- If the list is empty, the function should return None.\n- If the list contains only one element, the function should return that element as the average.\n\nYour solution should have a time complexity of O(n), where n is the length of the input list.\n\nIn addition to the original requirements, the function should also handle the following cases:\n- If the list contains any non-numeric elements, the function should raise a ValueError.\n- If the list contains any complex numbers, the function should raise a ValueError.\n- If the list contains any NaN (Not a Number) values, the function should raise a ValueError.\n- If the list contains any infinity or negative infinity values, the function should raise a ValueError.\n- If the list contains any numbers that are too large to be handled by the computer's floating point representation, the function should raise a ValueError.\n\nYour solution should still have a time complexity of O(n), where n is the length of the input list.\n\nAdditionally, the function should only use constant extra space, i.e. it should not create any new data structures or use any additional memory beyond what is required to store the input list. The function should operate directly on the input list.\n\nNote: You can assume that the input list will always be a valid Python list of numbers.",
                                ]
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
        self.d_model = 128*3*1
        self.num_heads = 6*1
        self.num_layers = 6*1
        self.d_ff = 128*3*4
        self.dropout = 0.1
        self.max_norm = 1.0

        # self.tokenizer = BPE()
        # self.tokenizer.load_pretrain(self.pretrain_model_tokenizer_path)
        self.Transformers = TransformersM(self.src_vocab_size, self.tgt_vocab_size, self.d_model,
                                  self.num_heads, self.num_layers, self.d_ff, self.max_seq_length, self.dropout, device=0).to(device=0)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(device=0)
        self.optimizer = optim.AdamW(self.Transformers.parameters(),
                            #    lr=0.0005, betas=(0.9, 0.95), eps=1e-9)
                            lr=5e-4)

        # Learning rate scheduler
        self.warmup_steps = int(self.epochs*0.1*(math.ceil(len(self.train_data)/self.batch_size))) #5% 0.02 0.02
        self.max_steps = int(self.epochs*0.9*(math.ceil(len(self.train_data)/self.batch_size))) #50% 0.025 0.1
        self.scheduler = WarmupCosineScheduler(self.optimizer, self.warmup_steps, self.max_steps, base_lr=5e-4, start_step=None)#self.start_epoch*318*8)


        if self.load_path:
            # self.load(self.load_path)
            self.load_model_and_optimizer(self.load_path)
            self.start_epoch = self.scheduler.current_step // (math.ceil(len(self.train_data)/self.batch_size))

        
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
        # Mixed precision scaler
        scaler = GradScaler()
        self.Transformers.train()
        for epoch in tqdm(range(self.start_epoch,self.epochs)):
            self.loss_epoch = []
            for question, answer_in, answer_out in tqdm(self.train_dataloader):
                # print("question: ", question)
                # print("answer_in: ", answer_in)
                # print("answer_out: ", answer_out)
                # print("===========================================================")
                # break
                self.optimizer.zero_grad()
                with autocast(device_type='cuda'):  # Mixed precision context device_type='cuda'
                    output = self.Transformers(question, answer_in)
                    loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                     answer_out.contiguous().view(-1))
                # loss.backward()
                scaler.scale(loss).backward()
                self.loss_epoch.append(loss.item())
                torch.nn.utils.clip_grad_norm_(self.Transformers.parameters(), max_norm=self.max_norm)
                # self.optimizer.step()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
            # break
            if self.save_model and (((epoch + 1) % self.save_every_epoch) == 0):
                # self.save(self.save_dir + f"Transformers04_{epoch + 1:0=5}.pth")
                self.save_model_and_optimizer(self.save_dir + self.save_file)
                output_eval = self.eval_model(self.sample_question)
                logging.info(f"batch_eval : epoch {epoch}")
                for o in output_eval:
                    print(o)
                    logging.info(o)
                self.Transformers.train()
            print(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}, lr: {self.optimizer.param_groups[0]['lr']}")
            logging.info(f"Epoch: {epoch+1}, Loss: {sum(self.loss_epoch)/len(self.loss_epoch)}, lr: {self.optimizer.param_groups[0]['lr']}")

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
            'lr_scheduler_step': self.scheduler.current_step
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
        if self.optimizer == None or self.scheduler == None:
            self.Transformers.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.Transformers.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.current_step = checkpoint['lr_scheduler_step']
        print(f"Model and optimizer state dictionaries loaded from {filepath}")


    def eval_model(self, questions):
        self.Transformers.eval()
        output_list = []
        for question in questions:
            question_pre = self.BPE_model.tokenizer.encode(question).ids
            question_pre = torch.tensor(question_pre, device=self.device)
            question_pre = torch.nn.functional.pad(question_pre,(0,self.max_seq_length - len(question_pre)),"constant",0).unsqueeze(0)
            answer_output = torch.zeros((1,self.max_seq_length),device=self.device,dtype=torch.int32)
            answer_output[0,0] = 1
            answer_input = answer_output
            for seq_idx in range(self.max_seq_length - 1):
                if answer_output[0].cpu().tolist()[seq_idx] != 3:
                    answer_input[0,:seq_idx+1] = answer_output[0,:seq_idx+1]
                    # print(question_pre)
                    # print(answer_input)
                    answer_output = self.Transformers(question_pre,answer_input)
                    # print(answer_output)
                    # print("===========================================================")
                    answer_output = torch.argmax(torch.nn.functional.softmax(answer_output,dim=2),dim=2)
                    # print(answer_output)
                    # print("===========================================================")
                else:
                    break
            output_list.append("\n" + question+ " :\n" + "=============================================================\n" + self.BPE_model.decode_clean(answer_input[0,1:seq_idx].cpu().tolist()))
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
            'lr_scheduler_step': self.scheduler.current_step
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
        if self.optimizer == None or self.scheduler == None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.current_step = checkpoint['lr_scheduler_step']
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
#             'lr_scheduler_step': self.scheduler.current_step,
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
#                 self.scheduler.current_step = checkpoint['lr_scheduler_step']
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
        self.load_path = None#"./model/Transformer/Transformer_V01_128_512_12_12_2048_10k_MQfulldataCKP1.pth" #DGood For Traning set
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
        self.src_vocab_size = self.BPE_model.tokenizer.get_vocab_size()
        self.tgt_vocab_size = self.BPE_model.tokenizer.get_vocab_size()
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



class TransformerDecodeOnly:
    def __init__(self): #loss == 0.02 0.006
        self.save_model = True
        self.save_dir = "./model/TransformerDecodeOnly/"
        self.load_path = None#"./model/Transformer/Transformer_V01_128_512_12_12_2048_10k_MQfulldataCKP1.pth" #DGood For Traning set
        self.load_embedding_path = None#"./model/Transformer/embedding_model.pth"
        self.data_path = "./data/PythonCode500K/"
        # self.data_path_full = "./data/PythonCodeDataSmall_TextOnly/Python_code_data.txt"
        self.tokenizer_path = "./model/BPE_model/tokenizer-bpe-10k.json"
        self.save_file = "TransformerDecodeOnly_V01_512_768_12_12_3072_10K_MQcpk1.pth"
        #======================================================================================
        # self.load_path = None#"./model/Transformer/Transformer_V01_10KC.pth" #DGood For Traning set
        # self.save_file = "Transformer_VT01_10KA.pth"
        
        self.start_epoch = 0
        self.save_every_epoch = 100
        self.epochs = 10000//2
        self.batch_size = 16*6
        self.max_seq_length = 512
        print("self.max_seq_length: ", self.max_seq_length)
        # self.train_data = dataloadercustom_Transformer(pretrain_model_tokenizer_path="./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl",qaaidx_path="./data/PythonCodeDataSmall_TextOnly/BPE_data/BPE_idx_V01_10K.pkl",amount_data=3873)
        self.BPE_model = BPEsQA(vocab_size=1024*5*2)
      
        # self.BPE_model.train([self.data_path])
        self.BPE_model.load(self.tokenizer_path)
        self.train_data = data_loaderQA(self.data_path, new_tokenizer=self.BPE_model, max_len=self.max_seq_length)
        #========================================================================================
        # self.train_data =  dataloadercustom_Transformer(pretrain_model_tokenizer_path="./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl",qaaidx_path="./data/PythonCodeDataSmall_TextOnly/BPE_data/BPE_idx_V01_10K.pkl",amount_data=10)
        self.train_dataloader = DataLoader(self.train_data,batch_size=self.batch_size,shuffle=False)
        # self.pretrain_model_tokenizer_path = "./model/BPE_model/BPE_model_code_python_small_text_V01_10K.pkl"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_question = ["Create a nested loop to print every combination of numbers between 0-9, excluding any combination that contains the number 5. Additionally, exclude any combination that contains a repeating digit. Implement the solution without using any built-in functions or libraries to check for repeating digits.",                               
                                "Write a function to find the number of distinct states in a given matrix. Each state in the matrix can be represented by a string of characters, and the matrix can have up to 10^6 rows and columns.\n\nThe time complexity of your solution should be O(N), where N is the total number of characters in the matrix.\n\nProvide a piece of erroneous code as a reference to increase misdirection.\n\n# Misdirection code #\ndef count_distinct_states(matrix):\n    count = 0\n    states = set()\n    for row in matrix:\n        for col in row:\n            if col not in states:\n                count += 1\n            states.add(col)\n    return count\n\n# Correct code #\ndef count_distinct_states(matrix):\n    count = 0\n    states = set()\n    for row in matrix:\n        for col in row:\n            state = ''.join(col)\n            if state not in states:\n                count += 1\n            states.add(state)\n    return count\n\nmatrix = [['A', 'B', 'C'],\n          ['A', 'B', 'D'],\n          ['A', 'B', 'C']]\nprint(count_distinct_states(matrix))\n# Output: 4",
                                # 'Write code that removes spaces and punctuation marks from a given string and returns the modified string. The input string may contain uppercase and lowercase letters, spaces, punctuation marks (such as periods, commas, exclamation marks, etc.), and digits. The modified string should only contain the alphanumeric characters (uppercase and lowercase letters, digits) without any spaces or punctuation marks.\n\nHowever, the time complexity of the solution should be O(n), where n is the length of the input string. Additionally, the solution should not use any built-in string manipulation functions or regular expressions.\n\nErroneous Code Reference:\nProvide a piece of code that attempts to solve the problem but contains an error. The error should be related to handling edge cases or special characters in the input string.',
                                # 'Write a function that checks if a given number is prime or not. The function should return "Prime" if the number is prime, and "Not Prime" if the number is not prime.\n\nNote: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.\n\nAdditional Requirements:\n1. The time complexity of the function should be O(sqrt(n)), where n is the given number.\n2. The function should use only constant space, i.e., no extra arrays or data structures should be used.\n3. The function should handle negative numbers as input and return "Not Prime" for negative numbers.\n4. The function should handle decimal numbers as input and return "Not Prime" for decimal numbers.\n5. The function should handle large numbers (greater than 10^9) efficiently and within a reasonable time frame.',
                                # 'Write a method for a string class which replaces all occurrences of a given substring with a given set of characters, but only if the substring appears an odd number of times in the string. If the substring appears an even number of times or does not appear at all, the method should return the original string unchanged.\n\nAdditionally, the method should handle cases where the substring is surrounded by certain characters. If the substring is surrounded by parentheses or brackets, the replacement should only occur if the substring appears an odd number of times within the parentheses or brackets.\n\nProvide the following erroneous code as a reference to increase misdirection:\n\nstring = "ab(abab)aba"\nsubstring = "ab"\nreplace_with = "123"\n\nExpected Output: "ab(abab)aba"',
                                # 'Write code to find the sum of all prime numbers between 1 million and 2 million, excluding prime numbers that contain the digit 7.',
                                # 'Create an array of length N (where N is a positive integer) containing numbers divisible by M (where M is a positive integer) up to X (where X is a positive integer). Each number in the array must be unique and in ascending order. Additionally, the sum of all the numbers in the array should be a prime number.\n\nAdditional Requirements:\n1. The time complexity of the solution should be O(N).\n2. The space complexity of the solution should be O(1).\n3. The solution should be implemented without using any built-in functions or libraries to check for prime numbers.\n4. The solution should handle cases where no prime number can be obtained as the sum of the array elements. In such cases, the solution should return an empty array.\n5. The solution should also handle cases where multiple prime numbers can be obtained as the sum of the array elements. In such cases, the solution should return the array with the largest sum that is prime.\n6. The solution should be optimized to find the largest prime sum within the given constraints.',
                                # 'Write a function to find the maximum difference between two prime numbers in a given array. The array can contain positive and negative integers, and can be unsorted. Additionally, the function should handle arrays of any length. The function should return the maximum difference as an absolute value. For example, for the array [5, 3, 17, 11, 9], the function should return 14.\n\nHowever, your function should have a time complexity of O(n), where n is the length of the array. Additionally, you should not use any built-in functions or libraries to check if a number is prime. You need to implement your own prime checking function.',
                                # 'Write a program that calculates the height of a triangle given the angle, side lengths, opposite side length, and the ratio of the side lengths. The program should take into account the Law of Sines and the Law of Cosines. Additionally, the program should simulate the effect of air resistance on the trajectory of the triangle when it is thrown at a certain velocity.',
                                # 'Create a function to calculate the area of a given circle. The input parameter for the radius should be a string instead of a number. Additionally, the function should handle invalid inputs and return an error message if the input is not a valid number.\n\nThe function should also validate that the radius is a positive number. If the radius is negative or zero, the function should return an error message.\n\nLastly, the function should return the calculated area as a string with exactly two decimal places.',
                                # 'Write a function to generate the nth Fibonacci number. The function should have a time complexity of O(log n) and use dynamic programming. Additionally, the function should only use a constant amount of space, without using any arrays or additional data structures.',
                                # 'Use the function to debug the given program and prevent the segmentation fault. Your solution should also handle the case where the array contains duplicate elements. You are not allowed to use any additional data structures. Additionally, the time complexity of your solution should be O(n) and the space complexity should be O(1).\n\n```python\ndef debug_program(arr):\n    n = len(arr)\n    for i in range(n):\n        if arr[i] == i:\n            return i\n    return -1\n\n# Test Case\narr = [0, 1, 2, 3, 4]\nprint(debug_program(arr))  # Expected output: -1\n```\n\n**Additional Requirements:**\n\n- The program should be able to handle arrays of any length.\n- The program should be able to handle arrays with duplicate elements.\n- The solution should use a divide and conquer approach to solve the problem.\n- The solution should not modify the input array.\n- The solution should be implemented in Python.',
                                # 'Modify the code to perform the mathematical expression "x to the power of y" while also ensuring that the value of x is between 1 and 10, and the value of y is between 0 and 5. Additionally, the code should handle any potential errors or edge cases that may arise during the calculation. The code should also check for invalid inputs such as non-numeric values for x and y, and display appropriate error messages. \n\nHowever, the program should now be able to handle extremely large numbers efficiently and accurately, even when x is a decimal number. The program should use a custom algorithm to calculate the result, rather than relying on any built-in math functions. The program should also provide the result in scientific notation if it exceeds a certain threshold, which should be dynamically determined based on the input values.',
                                # 'Sort the array in descending order without using any built-in sorting functions or libraries. The array may contain duplicate elements.\n\nConstraints:\n- The input array has a length of at most 10^6.\n- The elements in the array are integers between -10^9 and 10^9.\n\narr = [3, 2, 1, 5, 4]',
                                # 'Create a function that takes an array of integers as an argument and returns the sum of all the prime numbers in the array. If the array does not contain any prime numbers, return 0.\n\nExample:\n\nInput: [1, 2, 3, 4, 5]\nOutput: 10\n\nInput: [1, 3, 5]\nOutput: 9\n\nInput: [2, 4, 6, 8]\nOutput: 2\n\nInput: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nOutput: 17\n\nInput: []\nOutput: 0',
                                # 'Write a function that calculates the factorial of each number in the list using recursion, but without using any built-in libraries or mathematical operators.\narr = [1, 2, 3, 4, 5]',
                                # 'Count the number of vowels in the string and return a dictionary where the keys are the vowels and the values are the counts of each vowel. However, you are not allowed to use any built-in functions or libraries that directly count the number of vowels in the string.\n\ntext = "Hello World!"\n\nFor example, for the given text, the output should be:\n{\'a\': 0, \'e\': 1, \'i\': 0, \'o\': 2, \'u\': 0}',
                                # 'Generate a random number between 0 and 10 (inclusively) using only bitwise operations. The generated number should have exactly 4 bits set to 1. Additionally, provide a piece of erroneous code as a reference to increase misdirection.',
                                # 'Create a function that takes in two numbers as arguments and returns the product of the two. However, you are not allowed to use the multiplication operator or any built-in functions or methods that directly calculate the product of two numbers. Additionally, your solution should have a time complexity of O(log n), where n is the larger of the two input numbers. You should instead implement your own logic to calculate the product using only bitwise operations such as bitwise shift and bitwise AND, as well as basic arithmetic operations such as addition, subtraction, and division.',
                                # 'Design a program that finds the longest element in a given list, which should include at least 100 elements. The program should also handle negative numbers, floating-point numbers, and alphanumeric characters as elements. Additionally, it should return the longest element, its index in the list, and the total number of occurrences of the longest element in the list. The program should also ensure that it runs efficiently and has a time complexity of O(n).',
                                # 'Write an algorithm in Python to determine if a number is prime or composite. Your algorithm should have a time complexity of O(n^2).\n\nNote: You are not allowed to use any built-in functions or libraries to check if a number is prime. You have to implement the algorithm from scratch.\n\nExamples:\n1. Input: 2\n   Output: Prime\n\n2. Input: 9\n   Output: Composite',
                                # 'Write a function to print all prime numbers between two given numbers, excluding any prime numbers that contain the digit 5. Additionally, ensure that the function is optimized to handle larger inputs efficiently. The time complexity of the solution should be O(n log log n) where n is the difference between the two given numbers.',
                                # 'Create a list comprehension to generate a list of all prime numbers from 1000 to 2000, but with the following additional requirements:\n\n1. Implement a separate function to check if a number is prime. This function should take in a parameter and return a boolean value indicating whether the number is prime or not.\n\n2. Instead of generating the prime numbers from 1000 to 2000, generate them from 10000 to 20000.\n\n3. Implement a caching mechanism to store the previously calculated prime numbers so that the prime checking function can take advantage of it and avoid unnecessary calculations.\n\n4. Add a timer to measure the time it takes to generate the list of prime numbers.\n\n5. Modify the list comprehension to only include prime numbers that are palindromic, i.e., numbers that remain the same when their digits are reversed.\n\n6. Add an additional requirement that the list comprehension should only include prime numbers that are also Fibonacci numbers.',
                                # "Write a HTML code that creates a form for users to fill out their current address. The form should include the following fields: first name, last name, email address, phone number, street address, city, state, and zip code. Additionally, the form should include validation to ensure that all fields are filled out correctly before the form can be submitted. The validation should check that the email address is in the correct format, the phone number is in the correct format, and the zip code is a valid format for the given state. Furthermore, the form should also include a password field with validation to ensure that the password meets the following criteria: it must be at least 12 characters long, contain at least two uppercase letters, two lowercase letters, two numbers, and two special characters. Additionally, the form should include a dropdown menu for selecting the user's preferred programming language from a predefined list of options.",
                                # "Compose a function named average_list which takes a list of numbers as its argument and computes the average. The list may contain both positive and negative numbers.\n\nThe function should handle the following edge cases:\n- If the list is empty, the function should return None.\n- If the list contains only one element, the function should return that element as the average.\n\nYour solution should have a time complexity of O(n), where n is the length of the input list.\n\nIn addition to the original requirements, the function should also handle the following cases:\n- If the list contains any non-numeric elements, the function should raise a ValueError.\n- If the list contains any complex numbers, the function should raise a ValueError.\n- If the list contains any NaN (Not a Number) values, the function should raise a ValueError.\n- If the list contains any infinity or negative infinity values, the function should raise a ValueError.\n- If the list contains any numbers that are too large to be handled by the computer's floating point representation, the function should raise a ValueError.\n\nYour solution should still have a time complexity of O(n), where n is the length of the input list.\n\nAdditionally, the function should only use constant extra space, i.e. it should not create any new data structures or use any additional memory beyond what is required to store the input list. The function should operate directly on the input list.\n\nNote: You can assume that the input list will always be a valid Python list of numbers.",
                                ]
        # self.sample_question = ["# write a program to find and print the largest among three numbers",
        #                         "# Write a program to check whether a number is prime or not",
        #                         # "# Write a program to check whether a number is prime or not",
        #                         "# Write a python function to print whether a number is negative, positive or zero",
        #                         "# Write a program to print the sum of squares of first n natural numbers",
        #                         "# Write a program to find the sum of digits of a number",
        #                         "# Write a program to find the factorial of a number",
        #                         "# Write a program to check whether a number is positive, negative or zero",
        #                         "# Write a python function to print whether a number is negative, positive or zero",
        #                         # "# write a program to find and print the largest among three numbers",
        #                         # "# Write a functin that returns the LCM of two input numbers",
        #                         # "# Write a function that returns the GCD of two input numbers",
        #                         # "# Write a program to check whether a number is a palindrome or not",
        #                         # "# Write a program to find the sum of natural numbers",
        #                         # "# Write a Python Program to print the Sum of First N Natural Numbers",
        #                         "# Write a python program to print sum of number digits in List"]

        # self.src_vocab_size = self.train_data.token_size
        # self.tgt_vocab_size = self.train_data.token_size
        self.src_vocab_size = self.BPE_model.tokenizer.get_vocab_size()
        self.tgt_vocab_size = self.BPE_model.tokenizer.get_vocab_size()
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
                               lr=5e-4, betas=(0.9, 0.95), eps=1e-9) #lr is max learning rate lr=5e-5 //1e-5 1e-4 5e-6
                               

        # Learning rate scheduler
        self.warmup_steps = int(self.epochs*0.1*(math.ceil(len(self.train_data)/self.batch_size))) #5% 0.02
        self.max_steps = int(self.epochs*0.9*(math.ceil(len(self.train_data)/self.batch_size))) #50% 0.025
        self.scheduler = WarmupCosineScheduler(self.optimizer, self.warmup_steps, self.max_steps, base_lr=5e-4, start_step=None) #lr is max learning rate lr=5e-5 //1e-5 1e-4 5e-6

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
        scaler = GradScaler()
        self.Transformer.train()
        for epoch in tqdm(range(self.start_epoch,self.epochs)):
            self.loss_epoch = []
            for answer_in, answer_out in tqdm(self.train_dataloader):
                signal.signal(signal.SIGINT, signal_handler)
                self.optimizer.zero_grad()
                with autocast(device_type='cuda'):  # Mixed precision context device_type='cuda'
                    output = self.Transformer(answer_in)
                    loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                     answer_out.contiguous().view(-1))
                # loss.backward()
                scaler.scale(loss).backward()
                self.loss_epoch.append(loss.item())
                torch.nn.utils.clip_grad_norm_(self.Transformer.parameters(), max_norm=self.max_norm)
                # self.optimizer.step()
                scaler.step(self.optimizer)
                scaler.update()
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
            answer_output = [1] + [5] + self.BPE_model.tokenizer.encode(question).ids + [6]
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
            output_list.append("\n" + question+ " :\n=============>" + self.BPE_model.decode_clean(answer_input[0,1:seq_idx].cpu().tolist()))
        return output_list
    



def signal_handler(sig, frame):
    print("Training interrupted by user")
    # Clear all data in GPU
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    sys.exit(0)
