from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import csv
import time

def time_to_minutes(time_str):
    total_minutes = 0
    parts = time_str.split()
    for part in parts:
        if 'h' in part:
            total_minutes += int(part.strip('h')) * 60
        elif 'm' in part:
            total_minutes += int(part.strip('m'))
    return total_minutes

def clean_rating_amount(rate_amount_text):
    cleaned_text = rate_amount_text.replace('(', '').replace(')', '').replace(' ', '').replace('K', '').replace('k', '')
    if cleaned_text.replace('.', '').isdigit():  # Check if cleaned text is numeric with optional decimal point
        amount = float(cleaned_text)  # Convert to float to preserve decimal precision
        if 'K' in rate_amount_text.upper():  # Check for 'K' or 'k' (case insensitive)
            amount *= 1000
        return int(amount)  # Convert back to integer if no decimal places
    else:
        return 0

def scrape_imdb_data(url):
    # Setup Chrome WebDriver
    driver_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=driver_service)

    try:
        # Navigate to the IMDb movie chart page
        driver.get(url)
        time.sleep(5)  # Add a short delay to ensure page loads completely (adjust as needed)

        # Find elements for movie details
        movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/a/h3')
        movie_years = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[1]')
        movie_lengths = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[2]')
        movie_rates = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span')
        movie_rate_amounts = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span/span')
        movie_group = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[3]')

        min_length = min(len(movie_titles), len(movie_years), len(movie_rates), len(movie_lengths), len(movie_rate_amounts), len(movie_group))

        data = []
        for i in range(min_length):
            title = movie_titles[i].text
            year = movie_years[i].text
            rate_text = movie_rates[i].text.split('\n')[0]
            length = time_to_minutes(movie_lengths[i].text)
            rate_amount_text = movie_rate_amounts[i].text.strip()
            rate_amount = clean_rating_amount(rate_amount_text)
            group = movie_group[i].text.strip()

            data.append([title, year, rate_text, length, rate_amount, group])

        return data

    finally:
        # Quit the WebDriver session
        driver.quit()

def save_to_csv(data, filename):
    csv_filename = filename
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Title', 'Year', 'Rating', 'Duration (minutes)', 'Rating Amount', 'Group'])
        csvwriter.writerows(data)

    print(f"IMDb movie data saved to {csv_filename}")

if __name__ == "__main__":
    url = 'https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm'
    imdb_data = scrape_imdb_data(url)
    save_to_csv(imdb_data, 'imdb_movies.csv')
