from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

class SpeechToTextListener:
    def __init__(self):
        self.chrome_options = self.configure_chrome_options()
        self.driver = self.configure_chrome_driver(self.chrome_options)
        self.website_url = self.get_website_url()
        self.driver.get(self.website_url)

    def configure_chrome_options(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--use-fake-ui-for-media-stream")
        chrome_options.add_argument("--headless=new") 
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        return chrome_options
    
    def configure_chrome_driver(self, chrome_options):
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    def get_website_url(self):
        return "https://allorizenproject1.netlify.app/"

    def listen_continuously(self, prints: bool=False):
        try:
            start_button = self.wait_for_element_to_be_clickable(By.ID, 'startButton')
            start_button.click()
            print("Started listening... (say 'exit' to stop)")

            last_text = ""
            while True:
                try:
                    output_element = self.wait_for_element_to_be_present(By.ID, 'output', timeout=5)
                    current_text = output_element.text.strip()
                    
                    if current_text and current_text != last_text:
                        last_text = current_text
                        print(f"Recognized: {current_text}")
                        
                        
                        if "exit" in current_text.lower():
                            print("Exit command detected.")
                            break
                except Exception as e:
                    if prints:
                        print(f"[!] Error while fetching output: {e}")
                    time.sleep(1)
        finally:
            if hasattr(self, 'driver'):
                self.driver.quit()

    def wait_for_element_to_be_clickable(self, by, identifier, timeout=20):
        return WebDriverWait(self.driver, timeout).until(
            EC.element_to_be_clickable((by, identifier))
        )
    
    def wait_for_element_to_be_present(self, by, identifier, timeout=10):
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((by, identifier))
        )

if __name__ == "__main__":
    listener = SpeechToTextListener()
    listener.listen_continuously(prints=True)
