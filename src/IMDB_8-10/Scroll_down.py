from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def open_page_and_scroll(url):
    """
    Open a webpage, zoom out to 50%, scroll down, click 'Show More' to load more films,
    and stop when 500 movies are loaded.
    """
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        driver.get(url)
        time.sleep(3)

        # Zoom out the page to 50%
        zoom_percentage = 50  # Set zoom to 50%
        driver.execute_script(f"document.body.style.zoom='{zoom_percentage}%'")
        print(f"Page zoom set to {zoom_percentage}%.")

        # Accept cookies if present
        try:
            accept_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="__next"]/div/div/div[2]/div/button[2]'))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", accept_button)
            accept_button.click()
            print("Cookies accepted.")
            time.sleep(2)
        except Exception as e:
            print("No cookies acceptance button found:", e)

        # Define the 'Show More' button XPath
        show_more_button_xpath = '//*[@id="__next"]/main/div[2]/div[3]/section/section/div/section/section/div[2]/div/section/div[2]/div[2]/div[2]/div/span/button'

        # Initial film count
        films_loaded = 0
        target_films = 500

        # Scroll down the page
        while films_loaded < target_films:
            # Scroll down to the bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Give it some time to load new content

            try:
                # Wait for and click the 'Show More' button
                show_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, show_more_button_xpath))
                )
                print("Clicking 'Show More' button.")
                show_more_button.click()
                time.sleep(2)  # Allow new content to load

                # Update and print the number of films loaded
                films_loaded += 50
                print(f"Films loaded: {films_loaded}")
            except Exception as e:
                # If no "Show More" button, the page has reached the bottom
                print("No 'Show More' button found. Reached the bottom of the page.")
                break

        if films_loaded >= target_films:
            print(f"Target reached: {films_loaded} films loaded.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        driver.quit()

if __name__ == "__main__":
    url = 'https://www.imdb.com/search/title/?title_type=feature&user_rating=7,9.9&num_votes=10000,&sort=user_rating,desc'
    open_page_and_scroll(url)
