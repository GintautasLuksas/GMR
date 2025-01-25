import csv
import os
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("imdb_scraper.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger()

def zoom_out(driver, zoom_percentage=50):
    """Zoom out the page."""
    driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")
    logger.info(f"Page zoom set to {zoom_percentage}%.")

def accept_cookies(driver):
    """Accept cookies if the button is present."""
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

def load_more_movies(driver, show_more_button_xpath, target_films=4000):
    """Scroll and load more movies."""
    films_loaded = 0
    while films_loaded < target_films:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Allow new content to load

        try:
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, show_more_button_xpath))
            )
            logger.info("Clicking 'Show More' button.")
            show_more_button.click()
            time.sleep(2)  # Allow new content to load

            films_loaded += 50
            logger.info(f"Films loaded so far: {films_loaded}")
        except Exception as e:
            logger.info("No 'Show More' button found. Reached the bottom of the page.")
            break

    return films_loaded

def save_row_to_csv(row, filename='IMDB710_Additional.csv'):
    """
    Save a single row of data to a CSV file. Appends to the file if it exists.
    """
    file_exists = os.path.exists(filename)
    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:  # Write header only once
                header = ['Directors', 'Stars']
                writer.writerow(header)
            writer.writerow(row)
        logger.info(f"Row saved to {filename}: {row}")
    except Exception as e:
        logger.error("Error saving row to CSV: %s", e)

def scrape_additional_info(driver, output_file='IMDB710_Additional.csv'):
    """
    Scrape additional IMDb movie information: director and stars.
    Write each result directly to the CSV file.
    """
    info_buttons = driver.find_elements(By.XPATH, './/div[1]/div[3]/button')
    logger.info(f"Found {len(info_buttons)} movies to scrape.")
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

            # Scrape director information
            director_elements = driver.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[1]/ul/li/a')
            directors = ", ".join([elem.text for elem in director_elements])

            # Scrape star information
            star_elements = driver.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[2]/ul/li/a')
            stars = ", ".join([elem.text for elem in star_elements])

            # Save each row to CSV immediately
            save_row_to_csv([directors, stars], output_file)

            logger.info(f"Scraped and saved movie {i + 1}.")

            # Close the modal by pressing the ESC key
            ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error scraping movie {i + 1}: %s", e)
            # Proceed to the next movie button

def main():
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        url = 'https://www.imdb.com/search/title/?title_type=feature&user_rating=7,9.9&num_votes=10000,&sort=user_rating,desc'
        driver.get(url)
        time.sleep(3)

        # Zoom out and accept cookies
        zoom_out(driver)
        accept_cookies(driver)

        # Load 150 movies
        show_more_button_xpath = '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button'
        films_loaded = load_more_movies(driver, show_more_button_xpath, target_films=4000)
        logger.info(f"Total films loaded: {films_loaded}")

        # Go back to the top of the page
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)

        # Scrape additional information
        scrape_additional_info(driver, output_file='IMDB710_Additional.csv')

    finally:
        driver.quit()
        logger.info("Scraping process completed.")

if __name__ == "__main__":
    main()
