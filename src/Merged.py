import csv
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def zoom_out(driver, zoom_percentage=50):
    """
    Zoom out the page to the specified percentage.
    """
    driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")
    logger.info(f"Page zoom set to {zoom_percentage}%")


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
        time.sleep(2)
    except Exception as e:
        logger.warning("No cookies acceptance button found: %s", e)


def load_more_movies(driver, show_more_button_xpath, target_films=500):
    """
    Scrolls down and clicks 'Show More' to load movies until the target count is reached.
    Continuously loads more movies while counting up to the desired amount.
    """
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
            logger.info(f"Films loaded: {films_loaded}")
            if films_loaded >= target_films:
                break
        except Exception as e:
            logger.warning("No 'Show More' button found. Reached the bottom of the page.")
            break

    return films_loaded


def time_to_minutes(time_str):
    """
    Convert a time string (e.g., "2h 30m") to total minutes.
    Extracts hours and minutes from a string to convert it into total minutes.
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
    Removes unwanted characters and formats the number for ratings (K for thousands, M for millions).
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
    Strips leading numbers and dots from movie titles to clean them up.
    """
    import re
    return re.sub(r'^\d+\.\s*', '', title)


def save_to_csv(data, filename='IMDB710.csv'):
    """
    Save scraped data to a CSV file.
    Writes the collected movie data to a CSV file for future analysis.
    """
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Writing the header
            writer.writerow(
                ['Index', 'Title', 'Year', 'Rating', 'Length (minutes)', 'Rating Amount', 'Group', 'Metascore', 'Directors', 'Stars', 'Short Description'])
            # Writing the data rows
            for row in data:
                writer.writerow(row)
        logger.info(f"Data saved to {filename}.")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")


def scrape_additional_info(driver):
    """
    Scrape additional IMDb movie information: director and stars.
    """
    info_buttons = driver.find_elements(By.XPATH, './/div[1]/div[3]/button')
    data = []

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

            # Log and extract director information
            director_elements = driver.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[1]/ul/li/a')
            directors = ", ".join([elem.text for elem in director_elements])
            logger.info(f"Directors: {directors}")

            # Log and extract star information
            star_elements = driver.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[2]/ul/li/a')
            stars = ", ".join([elem.text for elem in star_elements])
            logger.info(f"Stars: {stars}")

            data.append([directors, stars])

            logger.info(f"Scraped movie {i + 1}")

            # Close the modal by pressing the ESC key
            ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)

            # After scraping a movie, scroll down slightly to load the next set of movies
            driver.execute_script("window.scrollBy(0, 200);")
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error scraping movie {i + 1}: %s", e)

    return data


def scrape_imdb_data(url, target_films=500):
    """
    Scrape IMDb movie data from the given URL, including Metascore and additional information like directors, stars, and short description.
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

        # Load the specified number of movies
        films_loaded = load_more_movies(driver, show_more_button_xpath, target_films=target_films)

        # Now that the movies are loaded, start scraping data
        movie_titles = driver.find_elements(By.XPATH, './/div/div/div/div[1]/div[2]/div[1]/a')
        movie_years = driver.find_elements(By.XPATH, './/div/div/div/div[1]/div[2]/div[2]/span[1]')
        movie_lengths = driver.find_elements(By.XPATH, './/div/div/div/div[1]/div[2]/div[2]/span[2]')
        movie_rates = driver.find_elements(By.XPATH, './/div/div/div/div[1]/div[2]/span/div/span/span[1]')
        movie_rate_amounts = driver.find_elements(By.XPATH, './/div/div/div/div[1]/div[2]/span/div/span/span[2]')
        movie_group = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[1]/div/div/div/div[1]/div[2]/div[2]/span[3]')
        metascores = driver.find_elements(By.XPATH, './/div/div/div/div[1]/div[2]/span/span/span[1]')
        short_descriptions = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[1]/div/div/div/div[2]/div/div')

        min_length = max(len(movie_titles), len(movie_years), len(movie_rates), len(movie_lengths),
                         len(movie_rate_amounts), len(movie_group), len(metascores), len(short_descriptions))

        data = []
        for i in range(min_length):
            title = remove_number_prefix(movie_titles[i].text) if i < len(movie_titles) else 'N/A'
            year = movie_years[i].text if i < len(movie_years) else 'N/A'
            rate_text = movie_rates[i].text if i < len(movie_rates) else 'N/A'
            length = time_to_minutes(movie_lengths[i].text) if i < len(movie_lengths) else 0
            rating_amount = clean_rating_amount(movie_rate_amounts[i].text) if i < len(movie_rate_amounts) else 0
            group = movie_group[i].text if i < len(movie_group) else 'N/A'
            metascore = metascores[i].text if i < len(metascores) else 'N/A'
            description = short_descriptions[i].text if i < len(short_descriptions) else 'N/A'

            data.append([i + 1, title, year, rate_text, length, rating_amount, group, metascore, description, description])

        save_to_csv(data)
        logger.info(f"Scraped {films_loaded} movies.")

    except Exception as e:
        logger.error("Error during scraping: %s", e)
    finally:
        driver.quit()
if __name__ == "__main__":
    url = "https://www.imdb.com/search/title/?title_type=feature&user_rating=7,9.9&num_votes=10000,&sort=user_rating,desc"
    target_films = 5  # You can change this to any number you'd like to scrape
    scrape_imdb_data(url, target_films)