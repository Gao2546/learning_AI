from flask import Flask, request, jsonify, send_from_directory
from pandas import options
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
import dotenv
from googlesearch import search
from utils.util import extract_excel_text, extract_pdf_text, extract_image_text, extract_docx_text, extract_pptx_text, extract_txt_file, encode_text_for_embedding, extract_xls_text, save_vector_to_db, search_similar_documents_by_chat, EditedFileSystem
import requests

from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch

TEXT_FILE_EXTENSIONS = ['.txt', '.pdf', '.docx', '.pptx', '.odt', '.rtf']

# Load environment variables from .env file
dotenv.load_dotenv()
file_system = EditedFileSystem()

# Add project root to sys.path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TextToImage.utils.node import *

app = Flask(__name__)

def clear_gpu():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("Cleared GPU memory.")


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
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-infobars")
    # options.add_argument("--disable-dev-shm-usage")

    # Check if running in Docker
    if os.environ.get("IS_DOCKER") == "true":
        print("Running in Docker, setting headless mode.")
        options.add_argument("--headless")  # สำคัญสำหรับ docker
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")  # แก้ปัญหา /dev/shm space
    options.add_argument("--disable-gpu")  # ป้องกันบางปัญหาใน Linux
    options.add_argument("--remote-debugging-port=9222")

    # service = Service('/usr/local/bin/chromedriver')
    # driver = webdriver.Chrome(options=options)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=options)
    # driver = webdriver.ChromiumEdge(options=options)
    # driver = webdriver.Firefox(options=options)
    return driver


def send_image_to_server(image_path, save_path_on_server):
    url = os.path.join(APP_URL,"api" ,"save_img")  # Replace with real IP
    print(f"Sending image to server at {url}")
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"save_path": save_path_on_server}
        response = requests.post(url, files=files, data=data)
    response.raise_for_status()
    return response.json()

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
    clear_gpu()
    prompt = request.json['prompt']
    img_url = request.json['img_url']
    prompts = re.split(r"[ ,]+", prompt)  # Splits on spaces and commas
    s_prompt = []
    for i in prompts:
        s_prompt.append(int(i))
    _, data_path, img_path = model.generate(prompt=s_prompt, size=28, img_url=img_url)
    res = send_image_to_server(data_path, img_path)
    print(f"Image sent to server: {res}")
    # Generate the model
    return jsonify({'result': f'The model has been generated {prompt}', 'data_path': img_path})

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
    clear_gpu()
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
    clear_gpu()
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
    clear_gpu()
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
    clear_gpu()
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
    try:
        with DDGS() as ddgs:
            results_tmp = []
            results = ddgs.text(query, max_results=int(max_results), region="th-th")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}\n{result['href']}\n")
                results_tmp.append(f"{i}. {result['title']}\n{result['href']}\n")
                out_text += f"{i}. {result['title']}\n{result['href']}\n\n"
            results = results_tmp
        sto = time.time()
        print("use DDGS")
    except Exception as e:
        results = list(search(term=query,num_results=int(max_results), lang="th", region="th", ssl_verify=True))
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}\n")
            out_text += f"{i}. {result}\n\n"
        sto = time.time()
        print(f"Error occurred: {e}")
    print(f'complete dT = {sto - st} Sec')
    results = "\n".join(results)
    return jsonify({'result': results})


@app.route('/process', methods=['POST'])
def process():
    clear_gpu()
    text = request.form.get('text', '')
    files = request.files.getlist('files')
    user_id = request.form.get('user_id', 'default_user')
    chat_history_id = request.form.get('chat_history_id', 'default_chat')

    print(len(files), "files")
    print(f"Received files: {[file.filename for file in files]}")

    extracted_texts = []

    for file in files:
        filename = file.filename.lower()
        file_text = ""

        if filename.endswith('.pdf'):
            file_text = extract_pdf_text(file)
        elif filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            file_text = extract_image_text(file)
        elif filename.endswith(('.docx','.doc','.odt','.rtf')):
            file_text = extract_docx_text(file)
        elif filename.endswith(('.pptx','.ppt')):
            file_text = extract_pptx_text(file)
        elif filename.endswith(('.xlsx','.xlsm')):
            file_text = extract_excel_text(file)
        elif filename.endswith('.xls'):
            file_text = extract_xls_text(file)

        else:
            # Everything else, attempt to read as text/code
            file_text = extract_txt_file(file)

        if not file_text.strip():
            print(f"Skipped file (empty or unsupported): {filename}")
            continue

        extracted_texts.append(file_text)
        data_vector = encode_text_for_embedding(file_text)
        save_vector_to_db(user_id, chat_history_id, filename, file_text, data_vector)

    combined_text = text + "\n" + "\n".join(extracted_texts)

    return jsonify({'reply': f'Processed input with {len(files)} file(s). Preview: ' + combined_text[:300]})


@app.route('/search_similar', methods=['POST'])
def search_similar_api():
    clear_gpu()
    data = request.get_json()
    
    query = data.get('query')
    user_id = data.get('user_id')
    chat_history_id = data.get('chat_history_id')
    top_k = int(data.get('top_k', 5))

    print(f"Searching for similar documents with query: {query}, user_id: {user_id}, chat_history_id: {chat_history_id}, top_k: {top_k}")

    if not query or not user_id or not chat_history_id:
        return jsonify({"error": "Missing required fields: query, user_id, chat_history_id"}), 400

    results = search_similar_documents_by_chat(
        query_text=query,
        user_id=int(user_id),
        chat_history_id=int(chat_history_id),
        top_k=top_k
    )

    return jsonify({"results": results})


# --- Helper for API responses ---
def _api_response(data, message="", status_code=200):
    """A consistent helper for creating JSON responses."""
    return jsonify({"data": data, "message": message}), status_code

# --- Helper to get required fields from request body ---
def _get_required_fields(data, *fields):
    """Checks for required fields in JSON data and returns them."""
    if not data:
        return None, _api_response(None, "Request body must be JSON.", 400)
    
    values = []
    for field in fields:
        value = data.get(field)
        if value is None:
            return None, _api_response(None, f"Missing required field in request body: '{field}'.", 400)
        values.append(value)
        
    return values, None

# --- File Listing (Remains GET as it doesn't target a specific resource) ---
@app.route('/files/list', methods=['GET'])
def api_list_files():
    """
    Lists all managed files.
    Example: GET /files/list
    """
    files, error = file_system.list_files()
    if error:
        return _api_response(None, error, 500)
    return _api_response({"files": files}, "Successfully listed files.")

# --- Consolidated Read Endpoint ---
@app.route('/files/read', methods=['POST'])
def api_read_file():
    """
    Reads content from a file. The action is determined by the fields provided.
    - To read all: {"file_name": "my_doc.txt"}
    - To read specific lines: {"file_name": "my_doc.txt", "start_line": 1, "end_line": 5}
    - To read from start to a line: {"file_name": "my_doc.txt", "end_line": 5}
    """
    data = request.get_json()
    (file_name,), error_response = _get_required_fields(data, 'file_name')
    if error_response:
        return error_response

    start_line = data.get('start_line')
    end_line = data.get('end_line')

    try:
        # Case 1: Read specific line range
        if start_line is not None and end_line is not None:
            lines, error = file_system.read_line(file_name, int(start_line), int(end_line))
            msg = f"Successfully read lines {start_line}-{end_line} from '{file_name}'."
        # Case 2: Read from start until a specific line
        elif end_line is not None:
            lines, error = file_system.read_start_until_line_n(file_name, int(end_line))
            msg = f"Successfully read from start to line {end_line} from '{file_name}'."
        # Case 3: Read the whole file
        else:
            lines, error = file_system.read_all(file_name)
            # read_all returns a list, join it for a single content string
            lines = {"content": "\n".join(lines)} if not error else None
            msg = f"Successfully read all content from '{file_name}'."

        if error:
            return _api_response(None, error, 404)
        return _api_response(lines, msg)

    except (TypeError, ValueError):
        return _api_response(None, "Invalid 'start_line' or 'end_line' parameters. Must be integers.", 400)

# --- Consolidated Edit Endpoint ---
@app.route('/files/edit', methods=['POST'])
def api_edit_file():
    """
    Edits or overwrites a file. The action is determined by the fields provided.
    - To edit specific lines: {"file_name": "my_doc.txt", "text": "new line content", "start_line": 2, "end_line": 2}
    - To overwrite the whole file: {"file_name": "my_doc.txt", "text": "all new content"}
    """
    data = request.get_json()
    (file_name, text), error_response = _get_required_fields(data, 'file_name', 'text')
    if error_response:
        return error_response

    start_line = data.get('start_line')
    end_line = data.get('end_line')

    try:
        # Case 1: Edit specific lines
        if start_line is not None and end_line is not None:
            error = file_system.edit_line(file_name, text, int(start_line), int(end_line))
            msg = f"Successfully edited lines {start_line}-{end_line} in '{file_name}'."
        # Case 2: Overwrite the entire file
        else:
            error = file_system.edit_all(file_name, text)
            msg = f"Successfully overwritten file '{file_name}'."

        if error:
            return _api_response(None, error, 400)
        return _api_response(None, msg)

    except (TypeError, ValueError):
        return _api_response(None, "'start_line' and 'end_line' must be integers.", 400)


# --- Consolidated Create Endpoint ---
@app.route('/files/create', methods=['POST'])
def api_create_file():
    """
    Creates a new file.
    - To create an empty file: {"file_name": "new_empty.txt"}
    - To create a file with content: {"file_name": "new_content.txt", "text": "initial content"}
    """
    data = request.get_json()
    (file_name,), error_response = _get_required_fields(data, 'file_name')
    if error_response:
        return error_response
        
    text = data.get('text')

    # Case 1: Create file with text (overwrites if it exists)
    if text is not None:
        error = file_system.create_new_file_and_text(file_name, text)
        msg = f"Successfully created file '{file_name}' with text."
        if error:
            return _api_response(None, error, 500) # Internal Server Error on create failure
    # Case 2: Create an empty file only (fails if it exists)
    else:
        error = file_system.create_new_file_only(file_name)
        msg = f"Successfully created empty file '{file_name}'."
        if error:
            return _api_response(None, error, 409) # 409 Conflict if file exists

    return _api_response(None, msg, 201)

# --- Delete Endpoint ---
@app.route('/files/delete', methods=['POST'])
def api_delete_file():
    """
    Deletes a file.
    - Body: {"file_name": "file_to_delete.txt"}
    """
    data = request.get_json()
    (file_name,), error_response = _get_required_fields(data, 'file_name')
    if error_response:
        return error_response

    error = file_system.delete_file(file_name)
    if error:
        return _api_response(None, error, 404)
    return _api_response(None, f"Successfully deleted file '{file_name}'.")

# --- File Download Endpoint ---
@app.route('/files/download', methods=['POST'])
def api_download_file():
    """
    Downloads a specific file.
    Note: Using POST for a download is non-standard for browsers but works for programmatic clients.
    - Body: {"file_name": "my_document.txt"}
    """
    data = request.get_json()
    (file_name,), error_response = _get_required_fields(data, 'file_name')
    if error_response:
        return error_response
        
    full_path = file_system._get_full_path(file_name)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        return _api_response(None, f"File '{file_name}' not found.", 404)

    return send_from_directory(file_system.base_dir, file_name, as_attachment=True)


# --- Folder Creation Endpoint ---
@app.route('/files/create_folder', methods=['POST'])
def api_create_folder():
    """
    Creates a new folder.
    - Body: {"folder_name": "new_folder_name"}
    """
    data = request.get_json()
    (folder_name,), error_response = _get_required_fields(data, 'folder_name')
    if error_response:
        return error_response

    error = file_system.create_folder(folder_name)
    if error:
        return _api_response(None, error, 409) # 409 Conflict if folder exists, or 500 for other errors
    return _api_response(None, f"Successfully created folder '{folder_name}'.", 201)

class GemmaEmbeddings(Embeddings):
    def __init__(self, model_name: str = "google/embeddinggemma-300m", quantized: bool = True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if quantized:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=bnb_config,
            )
        else:
            self.model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

        self.device = next(self.model.parameters()).device

    def _embed(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def embed_documents(self, texts):
        return self._embed(texts).tolist()

    def embed_query(self, text):
        return self._embed([text])[0].tolist()


if __name__ == '__main__':
    # path_keys = os.popen("find ../ -name '.key'").read().split("\n")[0]
    # with open(path_keys, "r") as f:
    #     key = f.read().strip()
    APP_URL = os.getenv("API_APP", "http://localhost:5000")
    api_key = os.getenv("OPENAI_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

    model_name = "google/embeddinggemma-300m"

    
    # Configure quantization (new method)
    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,              # or 
        load_in_4bit=True,
        llm_int8_threshold=6.0,
    )



    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2") # *** ***
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small") *** *** ***
    # embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v2-moe")
    # embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2") ## slowest and but efficient *** *** ***
    # embeddings = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")

    embeddings = GemmaEmbeddings(model_name="google/embeddinggemma-300m", quantized=True)
    
    # embeddings = HuggingFaceEmbeddings(model_name="voyageai/voyage-3.5-lite")
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