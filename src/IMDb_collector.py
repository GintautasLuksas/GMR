from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import csv
import time


def time_to_minutes(time_str):
    """Convert a time string (e.g., '2h 30m') to total minutes."""
    total_minutes = 0
    parts = time_str.split()
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.strip('h')) * 60
        elif 'm' in part:
            total_minutes += int(part.strip('m'))
    return total_minutes


def clean_rating_amount(rate_amount_text):
    """Clean and convert rating amount text to an integer."""
    cleaned_text = rate_amount_text.replace('(', '').replace(')', '').replace(' ', '').replace('K', '').replace('k', '')
    if cleaned_text.replace('.', '').isdigit():  # Check if cleaned text is numeric with optional decimal point
        amount = float(cleaned_text)  # Convert to float to preserve decimal precision
        if 'K' in rate_amount_text.upper():
            amount *= 1000
        return int(amount)  # Convert back to integer if no decimal places
    else:
        return 0


def scrape_imdb_data(url):
    """Scrape IMDb movie data from the given URL."""
    # Manually set the path to the ChromeDriver
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'  # Change this to your chromedriver path
    driver_service = Service(chromedriver_path)  # Use the Service with the manually specified path
    driver = webdriver.Chrome(service=driver_service)

    try:
        driver.get(url)
        time.sleep(10)  # Wait for the page to load

        # Updated XPath expressions to scrape all relevant data
        movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[1]/a')
        movie_years = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/span[1]')
        movie_lengths = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/span[2]')
        movie_rates = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span/span[1]')
        movie_rate_amounts = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span/span[2]')
        movie_group = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/span[3]')

        # Ensure we get the minimum length of the lists to prevent index errors
        min_length = max(len(movie_titles), len(movie_years), len(movie_rates), len(movie_lengths),
                         len(movie_rate_amounts), len(movie_group))

        data = []
        for i in range(min_length):
            title = movie_titles[i].text if i < len(movie_titles) else 'N/A'
            year = movie_years[i].text if i < len(movie_years) else 'N/A'
            rate_text = movie_rates[i].text if i < len(movie_rates) else 'N/A'
            length = time_to_minutes(movie_lengths[i].text) if i < len(movie_lengths) else 0
            rate_amount = clean_rating_amount(movie_rate_amounts[i].text) if i < len(movie_rate_amounts) else 0
            group = movie_group[i].text if i < len(movie_group) else 'N/A'

            data.append([title, year, rate_text, length, rate_amount, group])

        # Debugging: print the data to ensure it's scraped correctly
        print(f"Scraped Data: {data}")

        return data

    except Exception as e:
        print(f"Error during scraping: {e}")
        return []

    finally:
        driver.quit()


def save_to_csv(data, filename):
    """Save the scraped data to a CSV file."""
    if data:  # Check if data is non-empty before saving
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Title', 'Year', 'Rating', 'Duration (minutes)', 'Rating Amount', 'Group'])
            csvwriter.writerows(data)

        print(f"IMDb movie data saved to {filename}")
    else:
        print("No data to save.")


if __name__ == "__main__":
    url = 'https://www.imdb.com/chart/top/?ref_=chtmvm_ql_3'
    imdb_data = scrape_imdb_data(url)
    save_to_csv(imdb_data, 'imdb_movies.csv')
