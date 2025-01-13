import os
import sys
import subprocess
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import csv


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from IMDb_collector import time_to_minutes, clean_rating_amount, scrape_imdb_data, save_to_csv


@pytest.fixture(scope="module")
def driver():
    driver_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=driver_service)
    yield driver
    driver.quit()

def test_data_scraping(driver):
    url = 'https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm'
    driver.get(url)

    movie_titles = driver.find_elements(By.XPATH,
                                        '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/a/h3')
    movie_years = driver.find_elements(By.XPATH,
                                       '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[1]')
    movie_lengths = driver.find_elements(By.XPATH,
                                         '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[2]')
    movie_rates = driver.find_elements(By.XPATH,
                                       '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span')
    movie_rate_amounts = driver.find_elements(By.XPATH,
                                              '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span/span')
    movie_group = driver.find_elements(By.XPATH,
                                       '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[3]')

    min_length = min(len(movie_titles), len(movie_years), len(movie_rates), len(movie_lengths), len(movie_rate_amounts),
                     len(movie_group))

    assert min_length > 0

def test_data_cleaning_functions():
    assert time_to_minutes('1h 30m') == 90
    assert time_to_minutes('2h') == 120
    assert time_to_minutes('45m') == 45
    assert time_to_minutes('') == 0

    assert clean_rating_amount('(10K)') == 10000
    assert clean_rating_amount('500') == 500
    assert clean_rating_amount('1.5K') == 1500
    assert clean_rating_amount('') == 0

def test_csv_output(driver):
    url = 'https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm'
    driver.get(url)

    movie_titles = driver.find_elements(By.XPATH,
                                        '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[2]/a/h3')
    movie_years = driver.find_elements(By.XPATH,
                                       '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[1]')
    movie_lengths = driver.find_elements(By.XPATH,
                                         '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[2]')
    movie_rates = driver.find_elements(By.XPATH,
                                       '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span')
    movie_rate_amounts = driver.find_elements(By.XPATH,
                                              '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/span/div/span/span')
    movie_group = driver.find_elements(By.XPATH,
                                       '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[2]/div/div/div[3]/span[3]')

    min_length = min(len(movie_titles), len(movie_years), len(movie_rates), len(movie_lengths), len(movie_rate_amounts),
                     len(movie_group))

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

    csv_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/imdb_movies.csv'))
    save_to_csv(data, csv_filename)

    assert os.path.isfile(csv_filename)  # Check if CSV file is created

    with open(csv_filename, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        assert headers == ['Title', 'Year', 'Rating', 'Duration (minutes)', 'Rating Amount', 'Group']

def test_script_execution():
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src', 'IMDb_collector.py'))

    result = subprocess.run(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0

if __name__ == "__main__":
    pytest.main()
