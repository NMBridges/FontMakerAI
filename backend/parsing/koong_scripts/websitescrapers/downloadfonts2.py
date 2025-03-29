import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup
website_url = "https://www.fontspace.com/list/a"

chrome_driver_path = '/opt/homebrew/bin/chromedriver'  # Specify the path to your chromedriver

# Ensure the download directory exists
download_dir = "/Users/alexkoong/Desktop/downloaded_fonts_june28"
if not os.path.exists(download_dir):
    print("STOPPPP")
    os.makedirs(download_dir)

options = webdriver.ChromeOptions()
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "safebrowsing.enabled": True,
    "profile.default_content_settings.popups": 0,
    "directory_upgrade": True
}
options.add_experimental_option("prefs", prefs)

# Create a Service object with the path to chromedriver
service = Service(chrome_driver_path)

# Initialize the webdriver with the Service object and options
driver = webdriver.Chrome(service=service, options=options)

next_button_css = ".pager-link.pager-next"

# download_button = driver.find_element(By.CSS_SELECTOR, "div[class='downloadButtonElement'] a")



list_css = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


css_download = "body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(3) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1) > g:nth-child(1) > path:nth-child(3)"




try: 

    for num in list_css:

        try:
                driver.get(website_url)
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 300);")
                time.sleep(5)

                download_button = driver.find_element(By.CSS_SELECTOR, css_download)
                driver.execute_script("arguments[0].click();", download_button)
                time.sleep(3)
        except: 
            print("hello")
        finally:
                print(website_url)



except: 
    print("all done")










'''

body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(3) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(5) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(6) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(7) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(8) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)

last 
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(24) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)

body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(3) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(5) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(7) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(8) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)


body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(3) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(5) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(12) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)
body > div:nth-child(1) > main:nth-child(3) > div:nth-child(2) > div:nth-child(2) > div:nth-child(24) > div:nth-child(2) > div:nth-child(2) > a:nth-child(2) > svg:nth-child(1)

'''