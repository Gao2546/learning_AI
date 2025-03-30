from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

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
        page_source = driver.page_source

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract all elements (tags) into a list, excluding script, img, and link (CSS) tags
        elements = [
            element
            for element in soup.find_all()
            if element.name not in ['script', 'img', 'svg']
        ]

        # Close the browser
        driver.quit()

        return elements

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    url_to_open = "https://www.example.com"  # Replace with the URL you want to open
    elements = get_page_source_as_list(url_to_open)

    if elements:
        print("HTML Elements (excluding script, img, and link):")
        for element in elements:
            print(element)
            print("-" * 50)
    else:
        print("Failed to retrieve page source.")