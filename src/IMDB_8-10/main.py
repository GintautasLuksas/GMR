import time
import csv
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


def zoom_out(driver, zoom_percentage=50):
    driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")
    logger.info(f"Page zoom set to {zoom_percentage}%.")


def accept_cookies(driver):
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


def load_more_movies(driver, show_more_button_xpath, target_films=150):
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
            time.sleep(2)  # Allow new content to load

            films_loaded += 50
            logger.info(f"Films loaded: {films_loaded}")
        except Exception as e:
            logger.info("No 'Show More' button found. Reached the bottom of the page.")
            break

    return films_loaded


def time_to_minutes(time_str):
    '''Convert a time string (e.g., "2h 30m") to total minutes.'''
    total_minutes = 0
    parts = time_str.split()
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.strip('h')) * 60
        elif 'm' in part:
            total_minutes += int(part.strip('m'))
    return total_minutes


def clean_rating_amount(rate_amount_text):
    '''Clean and convert rating amount text to an integer.'''
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
    """Remove numeric prefixes like '1. ' or '1.' from movie titles."""
    return re.sub(r'^\d+\.\s*', '', title)


def scrape_main_info(driver, max_movies=5):
    '''Scrape main movie information: title, year, rating, metascore, group, and others.'''
    data = []
    for i in range(1, max_movies + 1):
        # Constructing the XPaths based on the movie index
        title_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[1]/a/h3"
        year_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[1]"
        rating_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/div/span/span[1]"
        short_desc_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[2]/div/div"
        metascore_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/span/span[1]"
        group_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[3]"
        length_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[2]"
        rate_amount_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/span/div/span/span[2]"

        try:
            title = driver.find_element(By.XPATH, title_xpath).text
            year = driver.find_element(By.XPATH, year_xpath).text
            rating = driver.find_element(By.XPATH, rating_xpath).text
            short_desc = driver.find_element(By.XPATH, short_desc_xpath).text

            # Scraping metascore and group with fallback to 'N/A'
            metascore = 'N/A'
            try:
                metascore_text = driver.find_element(By.XPATH, metascore_xpath).text.strip()
                if metascore_text.isdigit():
                    metascore = int(metascore_text)
            except Exception:
                pass  # If metascore is not found, it will remain 'N/A'

            group = 'N/A'
            try:
                group_text = driver.find_element(By.XPATH, group_xpath).text.strip()
                if group_text:
                    group = group_text
            except Exception:
                pass  # If group is not found, it will remain 'N/A'

            # Scrape movie length and rating amount
            length = driver.find_element(By.XPATH, length_xpath).text
            rate_amount = driver.find_element(By.XPATH, rate_amount_xpath).text

            # Clean length and rating amount data
            length_minutes = time_to_minutes(length)
            cleaned_rate_amount = clean_rating_amount(rate_amount)

            logger.info(
                f"Scraped movie {i}: {title}, {year}, Rating: {rating}, Metascore: {metascore}, Group: {group}, Length: {length_minutes} minutes, Rate Amount: {cleaned_rate_amount}")

            data.append([title, year, rating, metascore, group, short_desc, length_minutes, cleaned_rate_amount])

        except Exception as e:
            logger.error(f"Error scraping movie {i}: {e}")
            continue

    return data


def scrape_additional_info(driver, max_movies=5):
    """Scrape additional information: directors and stars."""
    data = []
    for i in range(1, max_movies + 1):  # Assume we're scraping 5 movies
        directors_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[1]/div[2]/div[2]/span[2]"
        stars_xpath = f"//*[@id='__next']/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li[{i}]/div/div/div/div[2]/div/div"

        try:
            directors = driver.find_element(By.XPATH, directors_xpath).text
            stars = driver.find_element(By.XPATH, stars_xpath).text

            logger.info(f"Scraped additional info for movie {i}: Directors: {directors}, Stars: {stars}")
            data.append([directors, stars])

        except Exception as e:
            logger.error(f"Error scraping additional info for movie {i}: {e}")
            continue

    return data


def save_to_csv(data, filename='Scraped_Movie_Data.csv'):
    """Save the collected data to a CSV file."""
    header = ['Title', 'Year', 'Rating', 'Metascore', 'Group', 'Short Description', 'Length (Minutes)', 'Rate Amount',
              'Directors', 'Stars']
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)
        logger.info(f"Data saved to {filename}")
    except Exception as e:
        logger.error("Error saving data to CSV: %s", e)


def main():
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    driver.get("https://www.imdb.com/search/title/?title_type=feature&user_rating=7,9.9&num_votes=10000,&sort=user_rating,desc")
    zoom_out(driver)
    accept_cookies(driver)

    show_more_button_xpath = '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[3]/button'

    # Scrape the main data first
    main_data = scrape_main_info(driver, max_movies=5)

    # Now, collect additional data after interacting with any required buttons
    additional_data = scrape_additional_info(driver, max_movies=5)

    # Combine both sets of data
    combined_data = [main + additional for main, additional in zip(main_data, additional_data)]

    # Save the combined data to a CSV
    save_to_csv(combined_data)

    driver.quit()


if __name__ == "__main__":
    main()
