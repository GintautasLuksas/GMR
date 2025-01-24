import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def zoom_out(driver, zoom_percentage=50):
    """
    Zoom out the page to the specified percentage.
    """
    driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")

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
        time.sleep(2)
    except Exception as e:
        pass

def load_more_movies(driver, show_more_button_xpath, target_films=5):
    """
    Scrolls down and clicks 'Show More' to load movies until the target count is reached.
    Continuously loads more movies while counting up to the desired amount.
    """
    movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
    films_loaded = len(movie_titles)

    while films_loaded < target_films:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        try:
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, show_more_button_xpath))
            )
            show_more_button.click()
            time.sleep(2)

            movie_titles = driver.find_elements(By.XPATH, '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/ul/li/div/div/div/div[1]/div[2]/div[1]/a/h3')
            films_loaded = len(movie_titles)
        except Exception as e:
            break

def additional_scrape(url):
    """
    Scrape additional IMDb data like description and reviews from the given URL.
    """
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        driver.get(url)
        time.sleep(4)

        zoom_out(driver)
        accept_cookies(driver)

        show_more_button_xpath = '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button'
        load_more_movies(driver, show_more_button_xpath, target_films=10)

    finally:
        driver.quit()

