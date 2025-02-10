import csv
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def zoom_out(driver, zoom_percentage=50):
    """Zooms out the page to the specified percentage."""
    driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")
    logger.info(f"Page zoom set to {zoom_percentage}%.")


def accept_cookies(driver):
    """Accepts cookies if the accept button is present on the page."""
    try:
        accept_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div/div/div[2]/div/button[2]'))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", accept_button)
        accept_button.click()
        logger.info("Cookies accepted.")
        time.sleep(2)
    except Exception as e:
        logger.warning("No cookies acceptance button found: %s", e)


def load_more_movies(driver, show_more_button_xpath, target_films=5):
    """Loads more movies by scrolling and clicking 'Show More' until the target count is reached."""
    movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
    films_loaded = len(movie_titles)

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

            movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
            films_loaded = len(movie_titles)
            logger.info(f"Films loaded: {films_loaded}")
        except Exception as e:
            logger.warning("No 'Show More' button found. Reached the bottom of the page.")
            break

    return films_loaded


def time_to_minutes(time_str):
    """Converts a time string (e.g., '2h 30m') to total minutes."""
    total_minutes = 0
    parts = time_str.split()
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.strip('h')) * 60
        elif 'm' in part:
            total_minutes += int(part.strip('m'))
    return total_minutes


def clean_rating_amount(rate_amount_text):
    """Cleans and converts rating amount text to an integer."""
    cleaned_text = rate_amount_text.replace('(', '').replace(')', '').replace(' ', '').replace('K', '').replace('k', '').replace('M', '').replace('m', '')
    if cleaned_text.replace('.', '').isdigit():
        amount = float(cleaned_text)
        if 'K' in rate_amount_text.upper():
            amount *= 1000
        elif 'M' in rate_amount_text.upper():
            amount *= 1000000
        return int(amount)
    return 0


def remove_number_prefix(title):
    """Removes numeric prefixes like '1. ' or '1.' from movie titles."""
    return re.sub(r'^\d+\.\s*', '', title)


def save_to_csv(data, filename='IMDB710.csv'):
    """Saves the collected movie data to a CSV file."""
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Title', 'Year', 'Rating', 'Length (minutes)', 'Rating Amount', 'Group', 'Metascore', 'Short Description'])
            for row in data:
                writer.writerow(row)
        logger.info(f"Data successfully saved to {filename}.")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")


def scrape_imdb_data(url, max_movies=5):
    """Scrapes IMDb movie data including title, year, rating, length, rating amount, group, metascore, and short description."""
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        logger.info(f"Opening URL: {url}")
        driver.get(url)
        time.sleep(4)

        zoom_out(driver)
        accept_cookies(driver)

        show_more_button_xpath = '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button'
        films_loaded = load_more_movies(driver, show_more_button_xpath, target_films=max_movies)

        data = []
        for i in range(1, max_movies + 1):
            title_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[1]/a/h3"
            metascore_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/span/span[1]"
            group_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[3]"
            year_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[1]"
            length_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[2]"
            rate_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/div/span/span[1]"
            rate_amount_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/div/span/span[2]"
            short_desc_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[2]/div/div"

            try:
                title = driver.find_element(By.XPATH, title_xpath).text
                year = driver.find_element(By.XPATH, year_xpath).text
                length = time_to_minutes(driver.find_element(By.XPATH, length_xpath).text)
                rate = driver.find_element(By.XPATH, rate_xpath).text
                rate_amount = clean_rating_amount(driver.find_element(By.XPATH, rate_amount_xpath).text)
                short_desc = driver.find_element(By.XPATH, short_desc_xpath).text

                data.append([i, title, year, rate, length, rate_amount, 'N/A', 'N/A', short_desc])
            except Exception as e:
                logger.error(f"Error scraping movie {i}: {e}")
                continue

        save_to_csv(data)
        return data
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return []


url = 'https://www.imdb.com/search/title/?title_type=feature&user_rating=7,9.9&num_votes=10000,&sort=user_rating,desc'
scraped_data = scrape_imdb_data(url, max_movies=4000)
