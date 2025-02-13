import csv
import time
import logging
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def zoom_out(driver, zoom_percentage=50):
    """Zoom out the page to the specified percentage."""
    driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")
    logger.info(f"Page zoom set to {zoom_percentage}%.")

def accept_cookies(driver):
    """Accept cookies if the button is present on the page."""
    try:
        accept_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div/div/div[2]/div/button[2]'))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", accept_button)
        accept_button.click()
        logger.info("Cookies accepted.")
    except Exception as e:
        logger.warning("No cookies acceptance button found: %s", e)

def load_more_movies_and_scrape(driver, show_more_button_xpath, target_films=3300, batch_size=50):
    """Scroll and load more movies until the target count is reached. Writes data to CSV in batches."""
    data = []
    last_scraped_index = 0
    movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
    films_loaded = len(movie_titles)
    for i in range(1, films_loaded + 1):
        try:
            data.extend(scrape_movie_data(driver, i))
            last_scraped_index = i
        except Exception as e:
            logger.error(f"Error scraping movie {i}: {e}")
            data.append([i, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
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
            for i in range(last_scraped_index + 1, films_loaded + 1):
                try:
                    data.extend(scrape_movie_data(driver, i))
                    last_scraped_index = i
                    if len(data) >= batch_size:
                        save_to_csv(data)
                        data = []
                except Exception as e:
                    logger.error(f"Error scraping movie {i}: {e}")
                    data.append([i, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
        except Exception:
            logger.warning("No 'Show More' button found or error occurred. Reached the bottom of the page.")
            break
    if data:
        try:
            save_to_csv(data)
        except Exception as e:
            logger.error(f"Error saving remaining data to CSV: {e}")
    return data

def scrape_movie_data(driver, i):
    """Scrape data for a specific movie given the index."""
    title_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[1]/a/h3"
    metascore_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/span/span[1]"
    try:
        title = driver.find_element(By.XPATH, title_xpath).text
        metascore = driver.find_element(By.XPATH, metascore_xpath).text.strip() if driver.find_elements(By.XPATH, metascore_xpath) else 'N/A'
        data_row = [i, remove_number_prefix(title), metascore]
    except Exception:
        data_row = [i, 'N/A', 'N/A']
    return [data_row]

def save_to_csv(data):
    """Appends the scraped movie data to a CSV file."""
    file_exists = os.path.isfile('imdb_movies.csv')
    headers = ['Index', 'Title', 'Metascore']
    with open('imdb_movies.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerows(data)
    logger.info("Data appended to imdb_movies.csv")

def remove_number_prefix(title):
    """Remove numeric prefixes from movie titles."""
    return re.sub(r'^\d+\.\s*', '', title)

def scrape_imdb_data(url, max_movies=3330):
    """Scrape IMDb movie data from the given URL."""
    options = Options()
    options.headless = True
    driver_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=driver_service, options=options)
    try:
        logger.info(f"Opening URL: {url}")
        driver.get(url)
        zoom_out(driver)
        accept_cookies(driver)
        show_more_button_xpath = '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button'
        data = load_more_movies_and_scrape(driver, show_more_button_xpath, target_films=max_movies)
        save_to_csv(data)
    finally:
        driver.quit()
url = 'https://www.imdb.com/search/title/?title_type=feature&user_rating=7,9.9&num_votes=10000,&sort=user_rating,desc'
scrape_imdb_data(url)
