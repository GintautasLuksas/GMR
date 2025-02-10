"""Scrapes IMDb movie information, including directors and stars, using Selenium."""

import csv
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def zoom_out(driver, zoom_percentage=50):
    """Sets the page zoom level."""
    driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")
    logger.info(f"Page zoom set to {zoom_percentage}%.")

def accept_cookies(driver):
    """Accepts website cookies if the button is present."""
    try:
        accept_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div/div/div[2]/div/button[2]'))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", accept_button)
        accept_button.click()
        logger.info("Cookies accepted.")
        time.sleep(2)
    except Exception as e:
        logger.error("No cookies acceptance button found: %s", e)

def load_more_movies(driver, show_more_button_xpath, target_films=4000):
    """Loads more movies by clicking the 'Show More' button until the target count is reached."""
    films_loaded = 0
    while films_loaded < target_films:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        try:
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, show_more_button_xpath))
            )
            logger.info("Clicking 'Show More' button.")
            show_more_button.click()
            time.sleep(2)
            films_loaded += 50
            logger.info(f"Films loaded: {films_loaded}")
        except Exception:
            logger.info("No 'Show More' button found. Reached the bottom of the page.")
            break
    return films_loaded

def click_element_if_in_view(driver, element):
    """Scrolls to an element and clicks it if possible."""
    try:
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        time.sleep(2)
        element.click()
        logger.info("Clicked on the element.")
    except Exception as e:
        logger.error("Error clicking element: %s", e)

def scrape_additional_info(driver, filename='IMDB710_Additional.csv'):
    """Scrapes additional IMDb movie information, including directors and stars."""
    info_buttons = driver.find_elements(By.XPATH, './/div[1]/div[3]/button')
    header = ['Directors', 'Stars']
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for i, button in enumerate(info_buttons):
                try:
                    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", button)
                    time.sleep(2)
                    retry_count = 3
                    while retry_count > 0:
                        try:
                            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(button)).click()
                            break
                        except Exception:
                            retry_count -= 1
                            time.sleep(1)
                    if retry_count == 0:
                        raise Exception("Failed to click the button after retries.")
                    time.sleep(3)
                    director_elements = driver.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[1]/ul/li/a')
                    directors = ", ".join([elem.text for elem in director_elements])
                    logger.info(f"Directors: {directors}")
                    star_elements = driver.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[2]/ul/li/a')
                    stars = ", ".join([elem.text for elem in star_elements])
                    logger.info(f"Stars: {stars}")
                    writer.writerows([[directors, stars]])
                    logger.info(f"Scraped movie {i + 1}")
                    ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                    time.sleep(1)
                    driver.execute_script("window.scrollBy(0, 200);")
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error scraping movie {i + 1}: %s", e)
    except Exception as e:
        logger.error("Error saving data to CSV: %s", e)

def main():
    """Initializes the web driver and runs the scraping process."""
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)
    try:
        url = 'https://www.imdb.com/search/title/?title_type=feature&user_rating=7,9.9&num_votes=10000,&sort=user_rating,desc'
        driver.get(url)
        time.sleep(2)
        zoom_out(driver)
        accept_cookies(driver)
        show_more_button_xpath = '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button'
        load_more_movies(driver, show_more_button_xpath, target_films=4000)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        scrape_additional_info(driver)
    finally:
        driver.quit()


main()