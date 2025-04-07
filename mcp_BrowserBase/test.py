from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
import time
import re


def get_page_source_as_list(url):
    """
    Opens a web page using Selenium, retrieves the page source, parses it with BeautifulSoup,
    and returns a list of HTML elements, excluding script, image, and CSS elements.

    Args:
        url (str): The URL of the web page to open.

    Returns:
        list: A list of BeautifulSoup Tag objects representing the HTML elements
              (excluding script, image, and CSS elements),
              or None if an error occurred.
    """
    try:
        # Set up the Chrome driver (ensure you have chromedriver installed and in your PATH)
        service = Service('/usr/local/bin/chromedriver')
        # service = Service() # Use default path for chromedriver
        # options = webdriver.ChromeOptions()
        options = webdriver.FirefoxOptions()
        # options.add_argument("--user-data-dir=/home/athip/.config/firefox/")  # Update this path
        options.add_argument("--user-data-dir=/home/athip/.cache/mozilla/firefox/")  # Update this path
        # options.add_argument("--user-data-dir=/home/athip/.config/google-chrome/")  # Update this path
        options.add_argument("--profile-directory=Default")  # Change to "Profile 1" if needed
        # You can add options like headless mode if needed:
        # options.add_argument("--headless")
        # driver = webdriver.Chrome(options=options)
        driver = webdriver.Firefox(options=options)

        # Open the web page
        driver.get(url)

        # Get the page source
        # page_source = driver.page_source
        page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
        print(driver.find_elements(By.CSS_SELECTOR, ".style-scope.ytmusic-carousel"))
        ta = "#suggestion-cell-0x0"
        print(driver.find_elements(By.CSS_SELECTOR, ta))
        ele = driver.find_elements(By.TAG_NAME, "yt-button-renderer")
        print("tttttttttttttttttttttttttttttttttt")
        print(ele)
        html = '''
<html>
  <body>
  <section>
    <div>
        <div>
            <div>
                <div>
                    <div>
                        <div>
                            <div>
                                <div></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    </section>
    <input type="text" />
    <input type="checkbox" />
    <input type="radio" />
    <input type="file" />
    <input type="button" />
    <input type="submit" />
    <input type="reset" />
    <input type="hidden" />
    <input type="image" />
    <input type="password" />
    <input type="search" />
    <input type="tel" />
    <input type="url" />
    <input type="email" />
    <input type="number" />
    <input type="color" />
    <input type="range" />
    <input type="datetime" />
    <input type="datetime-local" />
    <input type="month" />
    <input type="week" />
    <input type="time" />
    <input type="date" />
    <input type="text" />
    <input type="checkbox" />
    <input type="radio" />
    <input type="file" />
    <input type="button" />
    <input type="submit" />
    <yt-icon></yt-icon>
    <tp-yt-paper-slider>
      <div></div>
      <custom-element></custom-element>
    </tp-yt-paper-slider>
    <div><another-custom-tag /></div>
  </body>
</html>
'''

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # body_children = soup.find_all(recursive=False, limit=None, _stacklevel = 2)

        # Extract all elements (tags) into a list, excluding script, img, and link (CSS) tags
        # elements = [
        #     {"tag_name":element.name, "tag_attrs": element.attrs, "tag_text": element.text}
        #     for element in body_children
        #     if element.name not in ['script', 'img', 'svg', 'link', 'meta', 'head', 'noscript', 'style', 'span', 'path', 'section', 'g', 'ellipse', 'circle', 'rect', 'polygon', 'polyline', 'defs', 'title', 'text', 'iron-iconset-svg', 'use', 'stop', 'symbol', 'foreignobject', 'marker', 'lineargradient', 'radialgradient', 'filter', 'fegaussianblur', 'fecolormatrix', 'feBlend', 'feoffset', 'feMerge', 'femergeNode', 'feflood', 'fecomposite', 'mask', 'clippath', 'iframe', 'template', 'dom-if', 'dom-repeat', 'dom-bind', 'dom-module', 'dom-if-children', 'dom-repeat-children', 'dom-bind-children', 'dom-module-children', 'style-scope', 'style-scope-iron-iconset-svg', 'style-scope-iron-iconset-svg-children']
        #                     and    element.text.strip() != '' 
        # ]
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
                                    'style-scope-iron-iconset-svg-children', 'seonuxt_main']
            and (element.text != '')
            # AND ensure the element has some stripped text content OR it's an input/textarea/button etc.
            # and (element.get_text(strip=True) != '' or element.name in ['input', 'textarea', 'button', 'a', 'select', 'option']) # Keep form elements and links even if textless
        ]

        print(len(elements))
        # print(str(elements[0])[:4000])
        # for element in elements:
            # print(element.name)
        page_source = "\n".join(str(inp) for inp in elements)  # แปลงเป็นสตริง
        page_source = page_source.replace("\n", "")


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

        with open('page_source.txt', 'w', encoding='utf-8') as f:
            f.write(page_source)
        # print(page_source)

        time.sleep(3)  # Wait for 5 seconds to see the opened page
        # Close the browser
        driver.quit()
        return elements

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
from duckduckgo_search import DDGS

def search_duckduckgo(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results, region="th-th")
        print(results)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}\n{result['href']}\n")

if __name__ == "__main__":
    # emm = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    # embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    url_to_open = "https://music.youtube.com/search?q=%E0%B8%82%E0%B8%AD%E0%B9%83%E0%B8%AB%E0%B9%89%E0%B9%82%E0%B8%8A%E0%B8%84%E0%B9%80%E0%B8%A5%E0%B8%A7"  # Replace with the URL you want to open
    elements = get_page_source_as_list(url_to_open)

    # if elements:
    #     print("HTML Elements (excluding script, img, and link):")
    #     for element in elements:
    #         # print(element)
    #         print(len(element))
    #         print("-" * 50)
    #     print(len(elements))
    # else:
    #     print("Failed to retrieve page source.")
    # aa = "\n\n\ntest\n\n\n"
    # aa.replace("\n", "")
    # print(aa)

    # query = input("Enter your search query: ")
    # query = "google cloud storage"
    # search_duckduckgo(query)