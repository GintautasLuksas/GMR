import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException, TimeoutException


def remove_number_prefix(text):
    """Remove number prefix if it exists in the movie title."""
    return " ".join(text.split(" ")[1:])


def time_to_minutes(time_str):
    """Convert time string to minutes (e.g., '1h 30m' -> 90)."""
    minutes = 0
    if 'h' in time_str:
        hours = int(time_str.split('h')[0].strip())
        minutes += hours * 60
    if 'm' in time_str:
        mins = int(time_str.split('m')[0].split()[-1].strip())
        minutes += mins
    return minutes


def clean_rating_amount(rate_amount_text):
    '''Clean and convert rating amount text to an integer.'''
    cleaned_text = rate_amount_text.replace('(', '').replace(')', '').replace(' ', '').replace('K', '').replace('k', '').replace('M', '').replace('m', '')
    if cleaned_text.replace('.', '').isdigit():
        amount = float(cleaned_text)
        if 'K' in rate_amount_text.upper():
            amount *= 1000
        elif 'M' in rate_amount_text.upper():
            amount *= 1000000
        return int(amount)
    return 0


def scrape_imdb_data(url):
    """Scrape IMDb movie data along with additional information: short description, director, and stars."""
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        driver.get(url)

        # Wait for movie list elements to load
        WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located(
            (By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li')
        ))
        print("Page loaded successfully. Starting to scrape data...")

        # Scrape movie details
        movie_data = []
        movie_elements = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li')

        for i, movie in enumerate(movie_elements, start=1):
            try:
                title_elem = movie.find_element(By.XPATH, './/div/div/div/div[1]/div[2]/div[1]/a')
                year_elem = movie.find_element(By.XPATH, './/div/div/div/div[1]/div[2]/div[2]/span[1]')
                length_elem = movie.find_element(By.XPATH, './/div/div/div/div[1]/div[2]/div[2]/span[2]')
                rate_elem = movie.find_element(By.XPATH, './/div/div/div/div[1]/div[2]/span/div/span/span[1]')
                rate_amount_elem = movie.find_element(By.XPATH, './/div/div/div/div[1]/div[2]/span/div/span/span[2]')
                metascore_elem = movie.find_elements(By.XPATH, './/div/div/div/div[1]/div[2]/span/span/span[1]')
                info_button_elem = movie.find_element(By.XPATH, './/div[1]/div[3]/button')

                # Handle group information
                try:
                    group_elem = movie.find_element(By.XPATH, './/div/div/div/div[1]/div[2]/div[2]/span[3]')
                    group = group_elem.text
                except NoSuchElementException:
                    group = 'N/A'

                title = remove_number_prefix(title_elem.text)
                year = year_elem.text
                length = time_to_minutes(length_elem.text)
                rate_text = rate_elem.text
                rate_amount = clean_rating_amount(rate_amount_elem.text)

                # Check if metascore element exists
                if metascore_elem:
                    metascore = metascore_elem[0].text
                else:
                    metascore = 'N/A'

                # Click the info button to get additional details
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", info_button_elem)
                time.sleep(1)
                try:
                    info_button_elem.click()
                    time.sleep(2)

                    # Wait for 4 seconds before pressing ESC to exit
                    time.sleep(4)

                    # Press ESC key to exit the information view
                    actions = ActionChains(driver)
                    actions.send_keys(Keys.ESCAPE).perform()
                    time.sleep(1)
                except ElementClickInterceptedException:
                    print(f"Info button for movie {i} was not clickable. Skipping additional details.")

                # Scrape short description, director, and stars using updated XPaths
                try:
                    short_description = driver.find_element(By.XPATH, '//div[contains(@class, "ipc-html-content-inner-div")]').text
                except NoSuchElementException:
                    short_description = 'N/A'

                try:
                    # Updated XPath for Director
                    director_elements = driver.find_elements(By.XPATH,
                                                             '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[1]/ul')
                    directors = ", ".join([elem.text for elem in director_elements])

                    # Find star elements

                except NoSuchElementException:
                    directors = 'N/A'

                try:
                    # Updated XPath for Stars
                    star_elements = driver.find_elements(By.XPATH,
                                                         '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[2]/ul')
                    stars = ", ".join([elem.text for elem in star_elements])
                except NoSuchElementException:
                    stars = 'N/A'

            except (NoSuchElementException, TimeoutException) as e:
                print(f"Error retrieving data for movie {i}: {e}")
                title = year = length = rate_text = rate_amount = group = metascore = short_description = directors = stars = 'N/A'

            movie_data.append([i, title, year, rate_text, length, rate_amount, group, metascore, short_description, directors, stars])

        return movie_data

    except Exception as e:
        print(f"Error during scraping: {e}")
        return []

    finally:
        driver.quit()


def save_to_csv(movie_data, filename):
    """Save movie data to a CSV file."""
    header = ['ID', 'Title', 'Year', 'Rating', 'Length (min)', 'Rating Amount', 'Group', 'Metascore', 'Description', 'Director', 'Stars']
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(movie_data)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


def main():
    """Main function to scrape IMDb movie data and save it to CSV."""
    url = 'https://www.imdb.com/search/title/?title_type=feature&user_rating=8,10&num_votes=10000,&sort=user_rating,desc'
    movies_data = scrape_imdb_data(url)

    if movies_data:
        save_to_csv(movies_data, 'IMDB_810.csv')
    else:
        print("No movies found.")


if __name__ == '__main__':
    main()
