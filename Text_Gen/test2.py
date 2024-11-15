from util.util import BPE
import pandas as pd

texts = [
    "low low lower lowest",
    "newer new newest",
    "higher high highest",
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
    """In Python, variables are used to store information. Variables can be assigned to different types of data, including integers, floats, strings, and booleans. For example:
x = 10
y = 3.14
name = "Alice"
is_active = True
""",
    """Lists in Python are ordered, mutable collections. They allow duplicate elements. You can create a list and perform operations like accessing, adding, and removing elements. For example:
fruits = ["apple", "banana", "cherry"]
first_fruit = fruits[0]
fruits.append("date")
""",
    """Dictionaries in Python are collections of key-value pairs, defined using curly braces {}. Each key in a dictionary must be unique, and you can access values by their keys. Example:
person = {"name": "Alice", "age": 30, "city": "New York"}
name = person["name"]
person["email"] = "alice@example.com"
""",
    """Functions are reusable blocks of code that perform a specific task. They are defined with the def keyword, followed by the function name and parameters in parentheses. For example:
def greet(name):
    return f"Hello, {name}!"

message = greet("Alice")
print(message)
""",
    """If-else statements allow you to make decisions in your code. They execute code based on whether a condition is true or false. Example:
x = 20
if x > 10:
    print("x is greater than 10")
else:
    print("x is 10 or less")
""",
    """A loop allows you to repeat a block of code multiple times. For loops are commonly used to iterate over lists or ranges of numbers:
for i in range(5):
    print(i)
""",
]

data = pd.read_csv("./data/food_review_amazon/Reviews.csv", chunksize=10000)
if False:
    for d in data:
        print(d)
        d = d["Text"].tolist()
        texts = texts + d
        print(len(texts))

vocab_size = 1024*5
bpe = BPE()
path = "./model/BPE_model/bpe_test01.pkl"
# bpe.train(texts, vocab_size)
# output = bpe.tokenize("hello my name is athip!")
# print(output)
# bpe.save_model(path)
bpe.load_pretrain(path)
output = bpe.tokenize("hello my name is athip!")
print(output)
sentence = bpe.decode(output)
print(sentence)

# encoded = bpe.encode("newer lowest best")
# print("Encoded:", encoded)

# decoded = bpe.decode(encoded)
# print("Decoded:", decoded)

# print(bpe.word_freqs)

# print(bpe.vocab)
# print(len(bpe.vocab))

# print(bpe.splits)

# print("test".split())

["What are the differences between int, float, string, and bool in Python?",
 "How do you check the data type of a variable?",
 "Write a Python program to swap two variables.",
 "Explain the difference between == and =.",
 "Write a program that takes two numbers and prints their sum, difference, product, and quotient.",
 "Write a Python program that checks if a given number is positive, negative, or zero.",
 "Write a program to check if a year is a leap year or not.",
 "What is the difference between if and elif?",
 "Write a Python program to print numbers from 1 to 10 using a while loop.",
 "Explain the difference between for and while loops.",
 "Write a program to calculate the factorial of a number using a for loop."]
