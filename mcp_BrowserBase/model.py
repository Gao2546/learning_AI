from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import bs4
from langchain import hub
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from typing_extensions import List, TypedDict
# from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import time
import random
# from flask_sqlalchemy import SQLAlchemy
# from flask_marshmallow import Marshmallow
import os

app = Flask(__name__)

def init_driver():
    # Initialize the Chrome driver
    options = webdriver.FirefoxOptions()
    # options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    # options.add_argument('--headless')  # Run in headless mode
    # options.add_argument('--no-sandbox')
    # options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Firefox(options=options)
    return driver

def get_page(driver, url):
    # Get the page
    try :
        driver.get(url)
    except Exception as e:
        driver.get("http://duckduckgo.com")
    # Wait for the page to load
    driver.implicitly_wait(1)
    # Get the page source
    page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    return page_source

@app.route('/Generate', methods=['POST'])
def generate():
    text = request.json['text']
    # Generate the model
    return jsonify({'result': f'The model has been generated {text}'})

@app.route('/GetPage' , methods=['GET','POST'])
def get_page_route():
    global driver
    try:
        if driver:
            driver.quit()
    except:
        pass
    url = request.json['url']
    # sp = url.split("/")
    # if len(sp) > 3:
    #     url = "/".join(sp[:-1])
    driver = init_driver()
    driver.get(url)
    print("complete")
    return jsonify({'result': 'complete'})

@app.route('/GetSourcePage', methods=['GET','POST'])
def get_source_route():
    global vector_store
    global embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    soup = bs4.BeautifulSoup(page_source, "html.parser")
    elements = [
            element
            for element in soup.find_all()
            if element.name not in ['script', 'img']
        ]
    # inputs = soup.find_all(["input", "textarea"])  # ดึงเฉพาะ <input> elements
    page_source = "\n".join(str(inp) for inp in elements)  # แปลงเป็นสตริง
    pages = [str(inp) for inp in elements]
    _ = vector_store.add_texts(texts=pages) # Adds the split documents
    return jsonify({'page_source': page_source}) # Return the converted data

@app.route('/GetTextPage', methods=['GET','POST'])
def get_text():
    global vector_store
    global embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    soup = bs4.BeautifulSoup(page_source, "html.parser")
    soup_text = soup.get_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages = text_splitter.split_text(soup_text)
    _ = vector_store.add_texts(texts=pages)
    return jsonify({'page_source': page_source})

@app.route('/GetData', methods=['POST'])
def get_data():
    promt = request.json['prompt']
    k = request.json['k']
    vector_search = vector_store.similarity_search(promt, k=int(k))
    retrieved_docs = "\n\n".join(doc.page_content for doc in vector_search)
    print(type(retrieved_docs))
    print(len(retrieved_docs))
    app.logger.info("test")
    return jsonify({'retrieved_docs': retrieved_docs})

@app.route('/Search_By_ID', methods=['POST'])
def Search_By_ID():
    id = request.json['id']
    text = request.json['text']


    YTM_field = driver.find_elements(By.ID, id)
    if len(YTM_field) <= 0:
        YTM_field = driver.find_elements(By.CLASS_NAME, id)

    if YTM_field:
        for field in YTM_field:
           if field.is_displayed() and field.is_enabled():
                try:
                    field.send_keys(text)
                    break
                except:
                    pass
        # else:
            # print("Element is not visible")
            # return jsonify({"complete":"use another id or class from search list"})
    for char in "hello world":
        # input_box.send_keys(char)
        time.sleep(random.uniform(0.1, 0.3))
    field.send_keys(Keys.RETURN)
    return jsonify({"complete":"complete"})           



if __name__ == '__main__':
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = ""
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # vector_store = InMemoryVectorStore(embeddings)
    app.run(port=5001,debug=True)