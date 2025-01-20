from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import csv
import time

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
    '''Remove the numeric prefix (e.g., "1. ", "2. ") from movie titles.'''
    parts = title.split()
    if len(parts) > 1 and parts[0].isdigit() and parts[1] == '.':
        return ' '.join(parts[2:])
    return title

def scrape_imdb_data(url):
    '''Scrape IMDb movie data from the given URL.'''
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        driver.get(url)
        time.sleep(4)

        movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[1]/a')
        movie_years = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/span[1]')
        movie_lengths = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/span[2]')
        movie_rates = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span/span[1]')
        movie_rate_amounts = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span/span[2]')
        movie_group = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/span[3]')

        min_length = max(len(movie_titles), len(movie_years), len(movie_rates), len(movie_lengths), len(movie_rate_amounts), len(movie_group))

        data = []
        for i in range(min_length):
            title = remove_number_prefix(movie_titles[i].text) if i < len(movie_titles) else 'N/A'
            year = movie_years[i].text if i < len(movie_years) else 'N/A'
            rate_text = movie_rates[i].text if i < len(movie_rates) else 'N/A'
            length = time_to_minutes(movie_lengths[i].text) if i < len(movie_lengths) else 0
            rate_amount = clean_rating_amount(movie_rate_amounts[i].text) if i < len(movie_rate_amounts) else 0
            group = movie_group[i].text if i < len(movie_group) else 'N/A'

            data.append([i + 1, title, year, rate_text, length, rate_amount, group])

        return data

    except Exception as e:
        print(f"Error during scraping: {e}")
        return []

    finally:
        driver.quit()

def scrape_additional_info(url):
    '''Scrape additional IMDb movie information: short description, director, and stars.'''
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        driver.get(url)
        time.sleep(3)

        info_buttons = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[3]/button')

        data = []

        for i, button in enumerate(info_buttons):
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                time.sleep(1)

                retry_count = 3
                while retry_count > 0:
                    try:
                        button.click()
                        break
                    except Exception:
                        retry_count -= 1
                        time.sleep(1)

                if retry_count == 0:
                    raise Exception("Failed to click the button after retries.")

                time.sleep(3)

                short_description = driver.find_element(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[2]').text
                director_elements = driver.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[1]/ul/li')
                directors = ", ".join([elem.text for elem in director_elements])
                star_elements = driver.find_elements(By.XPATH, '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[2]/ul/li')
                stars = ", ".join([elem.text for elem in star_elements])

                data.append([short_description, directors, stars])

                ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(1)

            except Exception as e:
                print(f"Error scraping movie {i + 1}: {e}")

        return data

    finally:
        driver.quit()

def merge_data(imdb_data, additional_data):
    '''Merge IMDb data with additional information based on movie titles.'''
    merged_data = []
    for imdb_record, additional_record in zip(imdb_data, additional_data):
        merged_data.append(imdb_record + additional_record)
    return merged_data

def save_to_csv(data, filename):
    '''Save the merged data to a CSV file.'''
    if data:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['No.', 'Title', 'Year', 'Rating', 'Duration (minutes)', 'Rating Amount', 'Group', 'Short Description', 'Directors', 'Stars'])
            csvwriter.writerows(data)
        print(f"Merged IMDb movie data saved to {filename}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    url = 'https://www.imdb.com/chart/top/?ref_=chtmvm_ql_3'
    imdb_data = scrape_imdb_data(url)
    additional_data = scrape_additional_info(url)

    if imdb_data and additional_data:
        merged_data = merge_data(imdb_data, additional_data)
        save_to_csv(merged_data, 'IMDB_250.csv')
