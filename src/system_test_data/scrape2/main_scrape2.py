import csv
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import re
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def zoom_out(driver, zoom_percentage=50):
    """
    Zoom out the page to the specified percentage.
    """
    driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")
    logger.info(f"Page zoom set to {zoom_percentage}%.")


def accept_cookies(driver):
    """
    Accept cookies if the button is present on the page.
    Waits for the accept button to appear and clicks it.
    """
    try:
        accept_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div/div/div[2]/div/button[2]'))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", accept_button)
        accept_button.click()
        logger.info("Cookies accepted.")
    except Exception as e:
        logger.warning("No cookies acceptance button found: %s", e)


def load_more_movies_and_scrape(driver, show_more_button_xpath, target_films=50, batch_size=50):
    """
    Scrolls down and clicks 'Show More' to load movies until the target count is reached,
    and immediately scrapes the newly loaded movies after each click.
    Writes data to CSV every 'batch_size' movies and when errors occur.
    """
    data = []

    last_scraped_index = 0

    movie_titles = driver.find_elements(By.XPATH,
                                        '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
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


            movie_titles = driver.find_elements(By.XPATH,
                                                '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
            films_loaded = len(movie_titles)
            logger.info(f"Films loaded: {films_loaded}")


            for i in range(last_scraped_index + 1, films_loaded + 1):
                try:
                    data.extend(scrape_movie_data(driver, i))
                    last_scraped_index = i

                    # Write data to CSV after every 'batch_size' movies
                    if len(data) >= batch_size:
                        save_to_csv(data)
                        data = []  # Clear the data after writing to the CSV file

                except Exception as e:
                    logger.error(f"Error scraping movie {i}: {e}")
                    data.append(
                        [i, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])  # Append N/A for this movie

        except Exception as e:
            logger.warning("No 'Show More' button found or error occurred. Reached the bottom of the page.")
            break

    if data:
        try:
            save_to_csv(data)
        except Exception as e:
            logger.error(f"Error saving remaining data to CSV: {e}")

    return data


def scrape_movie_data(driver, i):
    """
    Scrape data for a specific movie given the index (i).
    """
    title_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[1]/a/h3"
    metascore_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/span/span[1]"
    group_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[3]"
    year_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[1]"
    length_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[2]"
    rate_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/div/span/span[1]"
    rate_amount_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/div/span/span[2]"
    short_desc_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[2]/div/div"

    data_row = []
    try:
        title = driver.find_element(By.XPATH, title_xpath).text
        year = driver.find_element(By.XPATH, year_xpath).text
        length = time_to_minutes(driver.find_element(By.XPATH, length_xpath).text)
        rate = driver.find_element(By.XPATH, rate_xpath).text
        rate_amount = clean_rating_amount(driver.find_element(By.XPATH, rate_amount_xpath).text)
        short_desc = driver.find_element(By.XPATH, short_desc_xpath).text

        logger.info(f"Scraping Movie {i}: {title}, Year: {year}, Rating: {rate}, Length: {length} min")

        metascore = 'N/A'
        try:
            metascore_text = driver.find_element(By.XPATH, metascore_xpath).text.strip()
            if metascore_text.isdigit():
                metascore = int(metascore_text)
        except:
            pass
        logger.info(f"Metascore for Movie {i}: {metascore}")

        group = 'N/A'
        try:
            group_text = driver.find_element(By.XPATH, group_xpath).text.strip()
            if group_text:
                group = group_text
        except:
            pass
        logger.info(f"Group for Movie {i}: {group}")

        data_row = [i, remove_number_prefix(title), year, rate, length, rate_amount, group, metascore, short_desc]

    except Exception as e:
        logger.error(f"Error scraping Movie {i}: {e}")
        data_row = [i, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']

    return [data_row]





def save_to_csv(data):
    """Appends the scraped movie data to a CSV file instead of overwriting it."""
    file_exists = os.path.isfile('imdb_movies2.csv')  # Check if the file already exists

    headers = ['Index', 'Title', 'Year', 'Rating', 'Length (mins)', 'Rating Amount', 'Metascore', 'Group', 'Short Description']

    with open('imdb_movies2.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(headers)

        writer.writerows(data)  # Append new data

    logger.info("Data appended to imdb_movies2.csv")



def time_to_minutes(time_str):
    """
    Convert a time string (e.g., "2h 30m") to total minutes.
    """
    total_minutes = 0
    parts = time_str.split()
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.strip('h')) * 60
        elif 'm' in part:
            total_minutes += int(part.strip('m'))
    return total_minutes


def clean_rating_amount(rate_amount_text):
    """
    Clean and convert rating amount text to an integer.
    """
    cleaned_text = rate_amount_text.replace('(', '').replace(')', '').replace(' ', '').replace('K', '').replace('k',
                                                                                                                '').replace(
        'M', '').replace('m', '')
    if cleaned_text.replace('.', '').isdigit():
        amount = float(cleaned_text)
        if 'K' in rate_amount_text.upper():
            amount *= 1000
        elif 'M' in rate_amount_text.upper():
            amount *= 1000000
        return int(amount)
    return 0


def remove_number_prefix(title):
    """
    Remove numeric prefixes like '1. ' or '1.' from movie titles.
    """
    return re.sub(r'^\d+\.\s*', '', title)


def scrape_imdb_data(url, max_movies=50):
    """
    Scrape IMDb movie data from the given URL, including Metascore and Short Description.
    Limits the number of movies to scrape.
    """
    options = Options()
    options.headless = True  # Run Chrome in headless mode to speed things up
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


# Call the scrape_imdb_data function with the IMDb URL
url = 'https://www.imdb.com/search/title/?release_date=2015-01-17,2024-12-31&num_votes=10000,'
scrape_imdb_data(url)
