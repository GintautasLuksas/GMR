import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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


def clean_rating_amount(rating_str):
    """Clean rating amount to a numeric value."""
    try:
        return int(rating_str.replace(',', '').strip())
    except ValueError:
        return 0


def scrape_imdb_data_810(url):
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
                group_elem = movie.find_element(By.XPATH, './/div/div/div/div[1]/div[2]/div[2]/span[3]')
                metascore_elem = movie.find_element(By.XPATH, './/div/div/div/div[1]/div[2]/span/span/span[1]')
                info_button_elem = movie.find_element(By.XPATH, './/div[1]/div[3]/button')

                title = remove_number_prefix(title_elem.text)
                year = year_elem.text
                length = time_to_minutes(length_elem.text)
                rate_text = rate_elem.text
                rate_amount = clean_rating_amount(rate_amount_elem.text)
                group = group_elem.text
                metascore = metascore_elem.text

                # Click the info button to get additional details
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", info_button_elem)
                time.sleep(1)
                info_button_elem.click()
                time.sleep(2)

                short_description = driver.find_element(By.XPATH, '//div[contains(@class, "ipc-html-content-inner-div")]').text
                directors = ", ".join([elem.text for elem in driver.find_elements(By.XPATH, '//div[contains(text(), "Director")]/following-sibling::ul/li')])
                stars = ", ".join([elem.text for elem in driver.find_elements(By.XPATH, '//div[contains(text(), "Stars")]/following-sibling::ul/li')])

                # Close the info dialog
                close_button = driver.find_element(By.XPATH, '//button[contains(@class, "ipc-close-button")]')
                close_button.click()
                time.sleep(1)

            except Exception as e:
                print(f"Error retrieving data for movie {i}: {e}")
                title = year = length = rate_text = rate_amount = group = metascore = short_description = directors = stars = 'N/A'

            movie_data.append([i, title, year, rate_text, length, rate_amount, group, metascore, short_description, directors, stars])

        return movie_data

    except Exception as e:
        print(f"Error during scraping: {e}")
        return []

    finally:
        driver.quit()


def main():
    """Main function to scrape IMDb movie data and additional information."""
    url = 'https://www.imdb.com/search/title/?title_type=feature&user_rating=8,10&num_votes=10000,&sort=user_rating,desc'
    movies_data = scrape_imdb_data_810(url)

    if movies_data:
        for movie in movies_data:
            print(f"Movie: {movie[1]}")
            print(f"Year: {movie[2]}")
            print(f"Rating: {movie[3]}")
            print(f"Length: {movie[4]} min")
            print(f"Rating Amount: {movie[5]}")
            print(f"Group: {movie[6]}")
            print(f"Metascore: {movie[7]}")
            print(f"Description: {movie[8]}")
            print(f"Director: {movie[9]}")
            print(f"Stars: {movie[10]}")
            print('-' * 60)
    else:
        print("No movies found.")


if __name__ == '__main__':
    main()
