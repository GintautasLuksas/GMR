import csv
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re


# Set up logging
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
        time.sleep(5)
    except Exception as e:
        logger.warning("No cookies acceptance button found: %s", e)

def load_more_movies(driver, show_more_button_xpath, target_films=4000):
    """
    Scrolls down and clicks 'Show More' to load movies until the target count is reached.
    Continuously loads more movies while counting up to the desired amount.
    """
    # Get the current count of movie titles in the view
    movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
    films_loaded = len(movie_titles)

    # If fewer movies are already loaded than the target, keep clicking 'Show More'
    while films_loaded < target_films:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Allow new content to load

        try:
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, show_more_button_xpath))
            )
            logger.info("Clicking 'Show More' button.")
            show_more_button.click()
            time.sleep(5)  # Allow new content to load

            # Get the updated count of movies
            movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
            films_loaded = len(movie_titles)
            logger.info(f"Films loaded: {films_loaded}")
        except Exception as e:
            logger.warning("No 'Show More' button found. Reached the bottom of the page.")
            break

    return films_loaded

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
    """
    Remove numeric prefixes like '1. ' or '1.' from movie titles.
    """
    return re.sub(r'^\d+\.\s*', '', title)

def save_to_csv(data, filename='IMDB710.csv'):
    """
    Save the collected movie data to a CSV file.
    """
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Writing the header
            writer.writerow(['Index', 'Title', 'Year', 'Rating', 'Length (minutes)', 'Rating Amount', 'Group', 'Metascore', 'Short Description'])
            # Writing the data rows
            for row in data:
                writer.writerow(row)
        logger.info(f"Data successfully saved to {filename}.")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")

# Updated XPaths for group and metascore
def scrape_imdb_data(url, max_movies=4000):
    """
    Scrape IMDb movie data from the given URL, including Metascore and Short Description.
    Limits the number of movies to scrape.
    """
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        logger.info(f"Opening URL: {url}")
        driver.get(url)
        time.sleep(4)

        # Zoom out and accept cookies
        zoom_out(driver)
        accept_cookies(driver)

        # Define the 'Show More' button XPath
        show_more_button_xpath = '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button'

        # Load up to 'max_movies' movies (e.g., 5)
        films_loaded = load_more_movies(driver, show_more_button_xpath, target_films=max_movies)

        # Now that the desired number of movies are loaded, start scraping data
        data = []
        for i in range(1, max_movies + 1):
            # Constructing the XPaths based on the movie index
            title_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[1]/a/h3"
            metascore_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/span/span[1]"
            group_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[3]"
            year_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[1]"
            length_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[2]"
            rate_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/div/span/span[1]"
            rate_amount_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/div/span/span[2]"
            short_desc_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[2]/div/div"

            # Scrape the data for each movie using the dynamically generated XPaths
            try:
                title = driver.find_element(By.XPATH, title_xpath).text
                year = driver.find_element(By.XPATH, year_xpath).text
                length = time_to_minutes(driver.find_element(By.XPATH, length_xpath).text)
                rate = driver.find_element(By.XPATH, rate_xpath).text
                rate_amount = clean_rating_amount(driver.find_element(By.XPATH, rate_amount_xpath).text)
                short_desc = driver.find_element(By.XPATH, short_desc_xpath).text

                # Log each movie's extracted data
                logger.info(f"Scraping Movie {i}: {title}, Year: {year}, Rating: {rate}, Length: {length} min")

                # Handle metascore and group, with N/A if missing
                metascore = 'N/A'
                try:
                    metascore_text = driver.find_element(By.XPATH, metascore_xpath).text.strip()
                    if metascore_text.isdigit():
                        metascore = int(metascore_text)
                except:
                    pass  # If metascore is not found, it will remain 'N/A'
                logger.info(f"Metascore for Movie {i}: {metascore}")

                group = 'N/A'
                try:
                    group_text = driver.find_element(By.XPATH, group_xpath).text.strip()
                    if group_text:
                        group = group_text
                except:
                    pass  # If group is not found, it will remain 'N/A'
                logger.info(f"Group for Movie {i}: {group}")

                # Append the data to the list
                data.append([i, title, year, rate, length, rate_amount, group, metascore, short_desc])

            except Exception as e:
                logger.error(f"Error scraping movie {i}: {e}")
                continue

        # Save the scraped data to a CSV file
        save_to_csv(data)
        return data

    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return []

# Example usage after scraping data
url = 'https://www.imdb.com/search/title/?title_type=feature&user_rating=7,9.9&num_votes=10000,&sort=user_rating,desc'  # Replace with the actual IMDb URL
scraped_data = scrape_imdb_data(url, max_movies=4000)