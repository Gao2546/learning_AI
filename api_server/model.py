from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import UnexpectedAlertPresentException, NoAlertPresentException
from selenium.webdriver.common.alert import Alert
import bs4
from langchain import hub
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from typing_extensions import List, TypedDict
# from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from duckduckgo_search import DDGS
import time
import random
# from flask_sqlalchemy import SQLAlchemy
# from flask_marshmallow import Marshmallow
import os
import sys
import re

# Add project root to sys.path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TextToImage.utils.node import *

app = Flask(__name__)

def init_driver():
    # Initialize the Chrome driver
    # options = webdriver.FirefoxOptions()
    options = webdriver.ChromeOptions()
    # options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    # options.add_argument('--headless')  # Run in headless mode
    # options.add_argument('--no-sandbox')
    # options.add_argument('--disable-dev-shm-usage')
    # options.add_argument("--user-data-dir=/home/athip/.cache/mozilla/firefox/")  # Update this path
    # options.add_argument("--user-data-dir=/home/athip/.config/google-chrome/")  # Update this path
    options.add_argument("--profile-directory=Default")  # Change to "Profile 1" if needed

    options.add_argument("--start-maximized")  # Open browser in full-screen
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    # service = Service('/usr/local/bin/chromedriver')
    driver = webdriver.Chrome(options=options)
    # driver = webdriver.ChromiumEdge(options=options)
    # driver = webdriver.Firefox(options=options)
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
    prompt = request.json['prompt']
    prompts = re.split(r"[ ,]+", prompt)  # Splits on spaces and commas
    s_prompt = []
    for i in prompts:
        s_prompt.append(int(i))
    model.generate(prompt=s_prompt, size=28)
    # Generate the model
    return jsonify({'result': f'The model has been generated {prompt}'})

@app.route('/GetPage' , methods=['GET','POST'])
def get_page_route():
    global driver
    st = time.time()
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
    sto = time.time()
    print(f'complete dT = {sto - st} Sec')
    return jsonify({'result': f'complete dT = {sto - st} Sec'})

@app.route('/Click' , methods=['GET','POST'])
def click_page_route():
    global driver
    st = time.time()
    id = str(request.json['Id'])
    classn = str(request.json['Class'])
    tag = str(request.json['TagName'])
    print(id)
    print(len(id))
    print(classn)
    print(len(classn))
    print(tag)
    print(len(tag))
    try:
        # YTM_field = driver.find_elements(By.ID, id)
        # if len(YTM_field) <= 0:
        #     YTM_field = driver.find_elements(By.CLASS_NAME, id)
        # elements = driver
        # if len(id) > 0:
        #     print("Please select the ID")
        #     elements = elements.find_element(By.ID, id)
        #     print("ID")
        # if len(classn) > 0:
        #     print("Please select the class")
        #     classn = classn.split(" ")[0]
        #     elements = elements.find_element(By.CLASS_NAME, classn)
        #     print("Class")
        # if len(tag) > 0:
        #     print("Please select the tag")
        #     elements = elements.find_element(By.TAG_NAME, tag)
        #     print("TagName")
        # elements = elements.find_all()
        classn = ".".join(classn.split(" "))
        print(f'{tag}{"#" + id if len(id) > 0 else ""}{"." + classn if len(classn) > 0 else ""}')
        elements = driver.find_elements(By.CSS_SELECTOR, f'{tag}{"#" + id if len(id) > 0 else ""}{"." + classn if len(classn) > 0 else ""}')
        


    except Exception as e:
        return jsonify({"result": str(e)})

    if elements:
        for field in elements:
           if field.is_displayed() or field.is_enabled():
                try:
                    field.click()
                    # field.send_keys(Keys.RETURN)
                    break
                except:
                    return jsonify({"result":"Element cannot be clicked"})
    else:
        print("Element not found")
        return jsonify({"result":"Element not found, use another id or class from search list"})
        # else:
            # print("Element is not visible")
            # return jsonify({"complete":"use another id or class from search list"})
    # for char in "hello world":
    #     # input_box.send_keys(char)
    #     time.sleep(random.uniform(0.1, 0.3))
    print("complete")
    sto = time.time()
    print(f'complete dT = {sto - st} Sec')
    return jsonify({'result': f'complete dT = {sto - st} Sec'})

@app.route('/GetSourcePage', methods=['GET','POST'])
def get_source_route():
    global vector_store
    # global embeddings
    st = time.time()
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = InMemoryVectorStore(embeddings)
    # vector_store.delete()
    # print(time.time() - st,"ssssssssssssssssssssssssssss")
    # page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    try:
        page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    except UnexpectedAlertPresentException:
        alert = Alert(driver)
        # print(f"Alert found: {alert.text}")
        # alert.accept()  # or alert.dismiss()
        # Optionally retry the operation
        page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    # print(time.time() - st,"ssssssssssssssssssssssssssss")
    soup = bs4.BeautifulSoup(page_source, "html.parser")
    # print(time.time() - st,"ssssssssssssssssssssssssssss")
    # elements = soup.find_all()
    # Find all elements recursively
    elements = [
            element
            # {"tag_name":element.name, "tag_attrs": element.attrs, "tag_text": element.text.replace("\n", "")}
            for element in soup.find_all(recursive=False) # Iterate through all elements recursively
            # Keep elements whose tag name is not in the exclusion list
            if element.name not in ['script', 'img', 'svg', 'head', 'link', 'meta', 'noscript', 
                                    'style', 'span', 'path', 'section', 'g', 'ellipse', 'circle', 
                                    'rect', 'polygon', 'polyline', 'defs', #'title', 'text', 
                                    'iron-iconset-svg', 'use', 'stop', 'symbol', 'foreignobject', 
                                    'marker', 'lineargradient', 'radialgradient', 'filter', 'fegaussianblur', 
                                    'fecolormatrix', 'feBlend', 'feoffset', 'feMerge', 'femergeNode', 
                                    'feflood', 'fecomposite', 'mask', 'clippath', 'iframe', 'template', 
                                    'dom-if', 'dom-repeat', 'dom-bind', 'dom-module', 'dom-if-children', 
                                    'dom-repeat-children', 'dom-bind-children', 'dom-module-children', 
                                    'style-scope', 'style-scope-iron-iconset-svg', 
                                    'style-scope-iron-iconset-svg-children']
            and (element.text != '')
            # AND ensure the element has some stripped text content OR it's an input/textarea/button etc.
            # and (element.get_text(strip=True) != '' or element.name in ['input', 'textarea', 'button', 'a', 'select', 'option']) # Keep form elements and links even if textless
        ]
    # print(time.time() - st,"ssssssssssssssssssssssssssss")
    # inputs = soup.find_all(["input", "textarea"])  # ดึงเฉพาะ <input> elements
    page_source = "\n".join(str(inp) for inp in elements)  # แปลงเป็นสตริง
    page_source.replace("\n", "")


    unwanted_tags = [
        'script', 'img', 'svg', 'head', 'link', 'meta', 'noscript',
        'style', 'span', 'path', 'section', 'g', 'ellipse', 'circle',
        'rect', 'polygon', 'polyline', 'defs', 'iron-iconset-svg', 'use',
        'stop', 'symbol', 'foreignobject', 'marker', 'lineargradient', 
        'radialgradient', 'filter', 'fegaussianblur', 'fecolormatrix',
        'feBlend', 'feoffset', 'feMerge', 'femergeNode', 'feflood', 
        'fecomposite', 'mask', 'clippath', 'iframe', 'template',
        'dom-if', 'dom-repeat', 'dom-bind', 'dom-module', 'dom-if-children',
        'dom-repeat-children', 'dom-bind-children', 'dom-module-children',
        'style-scope', 'style-scope-iron-iconset-svg',
        'style-scope-iron-iconset-svg-children'
    ]

    # Join into regex alternation group
    tag_pattern = '|'.join(unwanted_tags)

    # Pattern to remove full tags including content (for paired tags)
    full_tag_re = re.compile(rf'<(?:{tag_pattern})\b[^>]*>.*?</(?:{tag_pattern})>', re.DOTALL | re.IGNORECASE)
    remove_tags_re = re.compile(rf'</?({tag_pattern})\b[^>]*>', re.IGNORECASE)

    # Pattern to remove self-closing or opening tags (like <img ...> or <br/>)
    self_closing_re = re.compile(rf'<(?:{tag_pattern})\b[^>]*?/?>', re.IGNORECASE)

    # Pattern to remove closing tags (like </tag>)
    closing_tag_re = re.compile(rf'</(?:{tag_pattern})>', re.IGNORECASE)

    page_source = remove_tags_re.sub('', page_source)
    page_source = self_closing_re.sub('', page_source)
    page_source = closing_tag_re.sub('', page_source)
    # print(page_source)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    pages = text_splitter.split_text(page_source)
    # Convert selected elements (including their children) to strings
    # pages = [str(inp) for inp in elements]
    print(len(pages))
    _ = vector_store.add_texts(texts=pages)
    # print(time.time() - st,"ssssssssssssssssssssssssssss")
    sto = time.time()
    print(f'complete dT = {sto - st} Sec') # Return the converted data
    return jsonify({'result': f'complete dT = {sto - st} Sec'}) # Return the converted data

@app.route('/GetTextPage', methods=['GET','POST'])
def get_text():
    global vector_store
    # global embeddings
    st = time.time()
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = InMemoryVectorStore(embeddings)
    # vector_store.delete()
    # page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    try:
        page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    except UnexpectedAlertPresentException:
        alert = Alert(driver)
        print(f"Alert found: {alert.text}")
        alert.accept()  # or alert.dismiss()
        # Optionally retry the operation
        page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
    soup = bs4.BeautifulSoup(page_source, "html.parser")
    soup_text = soup.get_text()
    soup_text = soup_text.replace("\n", "")
    print(soup_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages = text_splitter.split_text(soup_text)
    _ = vector_store.add_texts(texts=pages)
    sto = time.time()
    print(f'complete dT = {sto - st} Sec') # Return the converted data
    return jsonify({'result': f'complete dT = {sto - st} Sec'})

@app.route('/GetData', methods=['POST'])
def get_data():
    st = time.time()
    promt = request.json['prompt']
    k = request.json['k']
    vector_search = vector_store.similarity_search(promt, k=int(k))
    retrieved_docs = "\n\n".join(doc.page_content for doc in vector_search)
    # print(type(retrieved_docs))
    # print(len(retrieved_docs))
    # app.logger.info("test")
    sto = time.time()
    print(f'complete dT = {sto - st} Sec')
    return jsonify({'retrieved_docs': retrieved_docs})

@app.route('/Search_By_ID', methods=['POST'])
def Search_By_ID():
    st = time.time()
    id = str(request.json['Id'])
    classn = str(request.json['Class'])
    tag = str(request.json['TagName'])
    text = request.json['text']
    try:
        classn = ".".join(classn.split(" "))
        print(f'{tag}{"#" + id if len(id) > 0 else ""}{"." + classn if len(classn) > 0 else ""}')
        elements = driver.find_elements(By.CSS_SELECTOR, f'{tag}{"#" + id if len(id) > 0 else ""}{"." + classn if len(classn) > 0 else ""}')
        
    except Exception as e:
        return jsonify({"result": str(e)})

    if elements:
        for field in elements:
           if field.is_displayed() or field.is_enabled():
                try:
                    field.send_keys(text)
                    field.send_keys(Keys.RETURN)
                    sto = time.time()
                    print(f'complete dT = {sto - st} Sec')
                    return jsonify({"result":f'complete dT = {sto - st} Sec'})  
                except:
                    return jsonify({"result":"Element cannot be clicked"})
    else:
        print("Element not found")
        return jsonify({"result":"Element not found, use another id or class from search list"})

    # try:
    #     YTM_field = driver.find_elements(By.ID, id)
    #     if len(YTM_field) <= 0:
    #         YTM_field = driver.find_elements(By.CLASS_NAME, id)

        # if YTM_field:
        #     for field in YTM_field:
        #        if field.is_displayed() and field.is_enabled():
        #             try:
        #                 field.send_keys(text)
        #                 break
        #             except:
        #                 pass
        #     # else:
        #         # print("Element is not visible")
        #         # return jsonify({"complete":"use another id or class from search list"})
        # # for char in "hello world":
        # #     # input_box.send_keys(char)
        # #     time.sleep(random.uniform(0.1, 0.3))
        # field.send_keys(Keys.RETURN)
    #     sto = time.time()
    #     print(f'complete dT = {sto - st} Sec')
    #     return jsonify({"result":f'complete dT = {sto - st} Sec'})     
    # except Exception as e:
    #     return jsonify({"result": str(e)})

@app.route('/Search_By_DuckDuckGo', methods=['POST'])     
def Search_By_DuckDuckGo():
    st = time.time()
    query = request.json['query']
    max_results = request.json['max_results']
    out_text = ""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=int(max_results), region="th-th")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}\n{result['href']}\n")
            out_text += f"{i}. {result['title']}\n{result['href']}\n\n"
    sto = time.time()
    print(f'complete dT = {sto - st} Sec')
    return jsonify({'result': results})


if __name__ == '__main__':
    # path_keys = os.popen("find ../ -name '.key'").read().split("\n")[0]
    # with open(path_keys, "r") as f:
    #     key = f.read().strip()
    key = os.environ.get("OPENAI_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2") # *** ***
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small") *** *** ***
    # embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v2-moe")
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2") ## slowest and but efficient *** *** ***
    # embeddings = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1") ##fastest but less efficient
    # embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2") ## fastest and efficient *** ***
    # embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L12-v2") ## faster and more efficient *** *** *
    # vector_store = InMemoryVectorStore(embeddings)
    image_c= 1
    img_size = 28
    model_ckp = "./TextToImage/model/checkpoint/DDPM_T0.pth"
    model_CLIP = "./TextToImage/model/checkpoint/CLIP0.pth"
    Text_dim = 512
    n_class = 10
    model = diffusion_model_No_VQVAE(
                in_c=image_c, 
                out_c=image_c,
                img_size=img_size,
                st_channel=64, 
                channel_multi=[1, 2, 4], 
                att_channel=64, 
                embedding_time_dim=64, 
                time_exp=256, 
                num_head=4, 
                d_model=32, 
                num_resbox=2, 
                allow_att=[True, True, True], 
                concat_up_down=True, 
                concat_all_resbox=True, 
                load_model_path=model_ckp,
                load_CLIP_path=model_CLIP,
                Text_dim=Text_dim,
                n_class=n_class

            )
    app.run(host='0.0.0.0', port=5000, debug=True)