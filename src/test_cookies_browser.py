import browser_cookie3
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
import time

# Constants
PAGE_LOAD_WAIT = 10  # Adjust the wait time as needed

# Load cookies from Brave
cj = browser_cookie3.brave()

# Initialize Selenium WebDriver
driver = webdriver.Firefox(service=FirefoxService())  # Ensure geckodriver is in your PATH

# Maximize and position the browser window
driver.set_window_position(0, 0)
driver.maximize_window()

# Open a page to set the domain for cookies
driver.get("https://www.instagram.com/aiandcivilization")

# Add cookies to the WebDriver
print("Adding cookies to the WebDriver...")
for cookie in cj:
    cookie_dict = {
        'name': cookie.name,
        'value': cookie.value,
        'domain': cookie.domain,
        'path': cookie.path,
        'expiry': int(cookie.expires) if cookie.expires else None,
        'secure': bool(cookie.secure),
        'httpOnly': bool(cookie.has_nonstandard_attr('HttpOnly'))
    }
    try:
        driver.add_cookie(cookie_dict)
    except Exception as e:
        #print(f"Error adding cookie: {cookie_dict['name']} - {e}")
        pass

print("Cookies added successfully !")

# Navigate to the target URL
url = "https://www.instagram.com/aiandcivilization"
driver.get(url)

# Wait for the page to load completely
time.sleep(PAGE_LOAD_WAIT)

# Print the page source
print(driver.page_source)

# Close the WebDriver
driver.quit()