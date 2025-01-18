from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import csv
import time


def scrape_additional_info(url):
    """Scrape additional IMDb movie information: short description, director, and stars."""
    # Set up the WebDriver
    chromedriver_path = r'D:\drivers\chromedriver-win64\chromedriver.exe'  # Adjust the path to your chromedriver
    driver_service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=driver_service)

    try:
        driver.get(url)
        time.sleep(5)  # Wait for the page to load

        # Locate all movie info buttons
        info_buttons = driver.find_elements(By.XPATH,
                                            '//*[@id="__next"]/main/div/div[3]/section/div/div[2]/div/ul/li/div[3]/button')

        print(f"Found {len(info_buttons)} movie buttons.")  # Debugging: confirm number of buttons found

        data = []

        for i, button in enumerate(info_buttons):
            try:
                print(f"Scraping movie {i + 1}...")  # Debugging: log current movie being scraped

                # Scroll to the button and click it
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                time.sleep(1)  # Add a slight delay after scrolling

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

                time.sleep(3)  # Wait for the modal to load

                # Scrape short description
                try:
                    short_description = driver.find_element(By.XPATH,
                                                            '/html/body/div[4]/div[2]/div/div[2]/div/div/div[2]').text
                except Exception:
                    short_description = "N/A"

                # Scrape director
                try:
                    director_elements = driver.find_elements(By.XPATH,
                                                             '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[1]/ul/li')
                    directors = ", ".join([elem.text for elem in director_elements])
                except Exception:
                    directors = "N/A"

                # Scrape stars
                try:
                    star_elements = driver.find_elements(By.XPATH,
                                                         '/html/body/div[4]/div[2]/div/div[2]/div/div/div[3]/div[2]/ul/li')
                    stars = ", ".join([elem.text for elem in star_elements])
                except Exception:
                    stars = "N/A"

                data.append([short_description, directors, stars])

                # Simulate pressing the ESC key to close the modal
                ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(1)  # Wait for the modal to close

            except Exception as e:
                print(f"Error scraping movie {i + 1}: {e}")

        return data

    finally:
        driver.quit()


def save_to_csv(data, filename):
    """Save the additional information to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Short Description', 'Directors', 'Stars'])
        csvwriter.writerows(data)

    print(f"Additional IMDb movie data saved to {filename}")


if __name__ == "__main__":
    url = 'https://www.imdb.com/chart/top/?ref_=chtmvm_ql_3'
    additional_data = scrape_additional_info(url)
    if additional_data:
        save_to_csv(additional_data, 'imdb_additional_info.csv')
