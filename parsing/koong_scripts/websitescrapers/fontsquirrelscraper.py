import csv

def csv_to_list_of_strings(file_path):
    """
    Reads a CSV file and converts it into a list of strings.

    :param file_path: Path to the CSV file
    :return: List of strings from the CSV file
    """
    result = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            for cell in row:
                result.append(cell.lower())
    
    return result

# Example usage:
file_path = '/Users/alexkoong/Desktop/scraped_data_fontsquirrel.csv'  # Replace with your CSV file path
list_of_fonts = csv_to_list_of_strings(file_path)



######################## csv parsing ##################


import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup
website_url = "https://www.1001freefonts.com/"

testing_url = "https://www.fontsquirrel.com/fonts/download/antic"

x=0

testing_list = [ 'salisbury-bold', 'playwrite-việt-nam', 'manosque']


url_end = '-font/'


chrome_driver_path = '/opt/homebrew/bin/chromedriver'  # Specify the path to your chromedriver

# Ensure the download directory exists
download_dir = "/Users/alexkoong/Desktop/downloaded_zips_fontsquirrel"
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



for item in list_of_fonts:

    website_url = f"https://www.fontsquirrel.com/fonts/download/{item}"


    try:
        time.sleep(2)
        driver.get(website_url)
        time.sleep(2)
        print(item)

    except: 
        print(KeyError)
    finally:
        print(website_url)


driver.quit()
