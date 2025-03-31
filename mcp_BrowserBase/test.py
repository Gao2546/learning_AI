from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings


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
        service = Service() # Use default path for chromedriver
        options = webdriver.ChromeOptions()
        # You can add options like headless mode if needed:
        # options.add_argument("--headless")
        driver = webdriver.Chrome(service=service, options=options)

        # Open the web page
        driver.get(url)

        # Get the page source
        # page_source = driver.page_source
        page_source = driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')


        # Extract all elements (tags) into a list, excluding script, img, and link (CSS) tags
        elements = [
            element
            for element in soup.find_all()
            if element.name not in ['script', 'img', 'svg', "link", 'meta','head','noscript','meta','style','div','span','path','section']
        ]

        print(len(elements))
        # for element in elements:
            # print(element.name)

        # Close the browser
        driver.quit()

        return elements

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # emm = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    url_to_open = "https://duckduckgo.com/?t=h_&q=%E0%B8%A7%E0%B8%B4%E0%B8%98%E0%B8%B5%E0%B8%97%E0%B8%B3%E0%B8%82%E0%B9%89%E0%B8%B2%E0%B8%A7%E0%B8%9C%E0%B8%B1%E0%B8%94&ia=web"  # Replace with the URL you want to open
    elements = get_page_source_as_list(url_to_open)

    if elements:
        print("HTML Elements (excluding script, img, and link):")
        for element in elements:
            print(element)
            print("-" * 50)
        print(len(elements))
    else:
        print("Failed to retrieve page source.")