from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
from datetime import datetime
from string import ascii_lowercase

import os


# Setup


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



'''

body > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(9) > div:nth-child(5) > a:nth-child(1) > strong:nth-child(1)
body > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(9) > div:nth-child(12) > a:nth-child(1) > strong:nth-child(1)
body > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(9) > div:nth-child(19) > a:nth-child(1) > strong:nth-child(1)
body > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(9) > div:nth-child(26) > a:nth-child(1) > strong:nth-child(1)
body > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(9) > div:nth-child(173) > a:nth-child(1) > strong:nth-child(1)

'''

css_selectors = []

for num in range(25): 
    number = (num * 7) + 5
    css_selectors.append(f"body > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(9) > div:nth-child({number}) > a:nth-child(1) > strong:nth-child(1)", 
)


# Function to scrape data for a single selector
def scrape_element(selector):
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return element.text
    except Exception as e:
        print(f"Error scraping selector '{selector}': {str(e)}")
        return "N/A"

# Function to scrape data for all selectors
def scrape_data(url):
    driver.get(url)
    
    data = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for i, selector in enumerate(css_selectors, 1):
        data[f'selector_{i}'] = scrape_element(selector)
    
    return data


# Function to save data to CSV
def save_to_csv(data, filename):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['date'] + [f'selector_{i}' for i in range(1, len(css_selectors) + 1)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        writer.writerow(data)

# Main script
try:
    for letter in ascii_lowercase:
        for number in range(385):

            page_num = number + 1

            scraped_data = scrape_data(f"https://www.dafont.com/alpha.php?lettre={letter}&page={page_num}")
            save_to_csv(scraped_data, 'scraped_data_fontsquirrel.csv')
            print("Data has been scraped and saved to scraped_data.csv")

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    driver.quit()