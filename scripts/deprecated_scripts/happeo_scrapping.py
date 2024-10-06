import argparse
import difflib
import json
import os
import pickle
import time

#from langchain_unstructured import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm


def is_already_scrapped(link, directory):
    """
    Check if a given link has already been scrapped.

    Args:
        link (str): The link to check.
        directory (str): The directory containing the scrapped pages.

    Returns:
        bool: True if the link has already been scrapped, False otherwise.
    """
    list_name_pages = os.listdir(directory)
    list_name_pages = [name.split(".")[0] for name in list_name_pages]
    for name in list_name_pages:
        if str(name) in str(link):
            return True
    return False


def load_cookies(driver):
    """
    Load cookies into the specified driver.

    Parameters:
    - driver: WebDriver object to load cookies into.

    Raises:
    - Exception: If there is an error loading the cookies.

    Prints:
    - "Trying to load cookies..." when attempting to load cookies.
    - "Cookies loaded successfully." when cookies are loaded successfully.
    - "Error loading cookies: <error_message>" when there is an error loading the cookies.
    - "No cookies found. Please log in manually." when no cookies are found.

    """
    try:
        print("Trying to load cookies...")
        cookies = pickle.load(open("./src/scrapping/cookies_happeo.pkl", "rb"))
        for cookie in cookies:
            driver.add_cookie(cookie)
        driver.refresh()
        print("Cookies loaded successfully.")
        time.sleep(2)
    except Exception as e:
        print("Error loading cookies: ", e)
        print("No cookies found. Please log in manually.")


def save_cookies(driver):
    """
    Save the cookies from the driver session.

    Parameters:
    - driver: WebDriver object

    Returns:
    None
    """
    cookies = driver.get_cookies()
    pickle.dump(cookies, open("./src/scrapping/cookies_happeo.pkl", "wb"))
    print("Cookies saved. You can now use them for future sessions.")


def fetch_links(driver):
    """
    Fetches all the links containing 'app.happeo' from the web page.

    Args:
        driver: The WebDriver instance used to interact with the web page.

    Returns:
        A list of cleaned links containing 'app.happeo' in the URL.

    """
  
    if is_dom_stable(driver):
        all_links = driver.find_elements(By.TAG_NAME, "a")
        list_href = [link.get_attribute("href") for link in all_links]
        clean_list_href = [link for link in list_href if link != None]

        # ne garde que les liens contenant 'app.happeo' dans la chaine de caractères
        clean_list_href = [
            link for link in clean_list_href if "app.happeo" in link
        ]

        return clean_list_href

def crawl_website(base_url, driver, directory):
    """
    Crawls a website starting from the given base URL using the provided driver.
    Args:
        base_url (str): The URL of the main page to start crawling from.
        driver: The driver object used to interact with the website.
        directory (str): The directory containing the scrapped pages.
    Returns:
        set: A set of unique links that have been visited during the crawling process.
    """

    visited = set()  # Pour éviter les répétitions

    print("STARTING LEVEL 1 CRAWLING")
    # Niveau 1: récupérer les liens de la page principale
    driver.get(base_url)
    main_page_links = fetch_links(driver)
    visited.update(main_page_links)
    print("Main page links fetched.")
    print("MAIN PAGE LINKS:", main_page_links)

    print("STARTING LEVEL 2 CRAWLING")

    # Niveau 2: Pour chaque lien sur la page principale, récupérer les liens des sous-pages
    for link in tqdm(main_page_links, desc="Visiting main page links"):
        # on ne vas sur le lien que si il contient 'happeo' dans la chaine de caractères
        try:
            driver.get(link)
            sub_page_links = fetch_links(driver)
            visited.update(
                sub_page_links
            )  # Ajoute les nouveaux liens à l'ensemble des liens visités
        except Exception as e:
            print(f"Erreur lors de l'accès à {link}: {e}")

    final_links = set()

    print("STARTING LEVEL 3 CRAWLING")

    # Niveau 3: Pour chaque lien sur les sous-pages, récupérer les liens des sous-sous-pages
    for link in tqdm(visited, desc="Visiting sub-page links"):
        try:
            driver.get(link)
            sub_sub_page_links = fetch_links(driver)
            final_links.update(
                sub_sub_page_links
            )  # Ajoute les nouveaux liens à l'ensemble des liens visités
        except Exception as e:
            print(f"Erreur lors de l'accès à {link}: {e}")

    # Print the size of the visited links list
    print("Visited links size:", len(visited))

    visited = final_links

    # delete all the None links
    visited = [link for link in visited if link != None]

    # Delete all the links that are already scrapped
    visited = [link for link in visited if not is_already_scrapped(link, directory)]

    print("FULL CLEANED VISITED LINKS:", visited)
    print("TOTAL NULBER OF UNIQUE LINKS NOT ALREADY SCRAPPED:", len(visited))

    return visited


def main():
    parser = argparse.ArgumentParser(description="Scrape Happeo pages.")
    parser.add_argument(
        "--directory",
        type=str,
        default="data/pages",
        help="Directory to save the scraped pages (default: data/pages)"
    )
    args = parser.parse_args()
    directory = args.directory

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox") #essential !
    chrome_options.add_argument("--disable-dev-shm-usage") #essential !
    chrome_options.add_argument("--remote-debugging-port=9222") #essential !
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1200")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://app.happeo.com/home")

    no_cookies = False
    try:
        load_cookies(driver)
    except:
        print("No cookies found. Please log in manually.")
        no_cookies = True
        pass

    time.sleep(5)
    save_cookies(driver)

    visited_links = crawl_website("https://app.happeo.com/home", driver, directory)
    true_links = 0
    for i, link in tqdm(enumerate(visited_links), total=len(visited_links)):
        if link != None:
            print(f"Processing link {i+1}/{len(visited_links)}")
            print("Current link:", link)
            save_page(driver, link, directory)
            time.sleep(1)
            true_links += 1

    driver.quit()

def is_dom_stable(driver, interval=0.5, attempts=5):
    """
    Check if the DOM (Document Object Model) is stable.
    Args:
        driver: The WebDriver instance to use for finding elements.
        interval (float, optional): The interval in seconds to wait between attempts. Defaults to 0.5.
        attempts (int, optional): The number of attempts to check the stability of the DOM. Defaults to 5.
    Returns:
        bool: True if the DOM is stable, False otherwise.
    """
 
    prev_count = 0
    for _ in range(attempts):
        # Attendre l'intervalle spécifié
        time.sleep(interval)
        # Compter le nombre d'éléments dans le DOM
        current_count = len(driver.find_elements(By.XPATH, "//*"))
        if current_count == prev_count:
            return True
        prev_count = current_count
    return False


def get_widgets(driver, link):
    """
    Retrieves the text content of all widgets on a webpage.

    Args:
        driver (WebDriver): The WebDriver instance used to interact with the webpage.
        link (str): The URL of the webpage.

    Returns:
        str: The concatenated text content of all widgets.

    Raises:
        None

    """
    driver.get(link)
    # load_cookies(driver) Inutile car on a déjà chargé les cookies !!!
    if is_dom_stable(driver):
        div_widget_elements = driver.find_elements(
            By.CSS_SELECTOR, "div.widget"
        )
        all_elements = []
        for widget in div_widget_elements:
            all_elements.extend(widget.find_elements(By.CSS_SELECTOR, "span"))
            all_elements.extend(widget.find_elements(By.CSS_SELECTOR, "strong"))
            all_elements.extend(widget.find_elements(By.CSS_SELECTOR, "a"))
            all_elements.extend(widget.find_elements(By.CSS_SELECTOR, "b"))
            all_elements.extend(widget.find_elements(By.CSS_SELECTOR, "p"))
        all_texts = [element.text for element in all_elements]
        return " ".join(all_texts)
    else:
        print(f"DOM not stable after retries: {link}")
        return None
def save_pages(driver, links, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for link in links:
        try:
            save_page(driver, link, directory)
        except Exception as e:
            print(f"Error saving page {link}: {e}")
            continue


def save_page(driver, link, directory):
    try:
        page_name = link.split("/")[-1] if link.split("/")[-1] else "index"
        widgets_text = get_widgets(driver, link)
        print(f"Widgets text for {page_name}: {widgets_text}")

        with open(
            f"{directory}/{page_name}.html", "w", encoding="utf-8"
        ) as f:
            f.write(driver.page_source)

        if widgets_text is not None:
            with open(
                f"{directory}/{page_name}_widgets.html",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"<div>{widgets_text}</div>")
        print(f"Page {page_name} saved successfully.")

        dict_file_path = f"{directory}/url_correspondance.json"
        if os.path.exists(dict_file_path):
            with open(dict_file_path, "r", encoding="utf-8") as dict_file:
                data = json.load(dict_file)

            data[str(page_name)] = link
            with open(dict_file_path, "w", encoding="utf-8") as dict_file:
                json.dump(data, dict_file)

        else:
            with open(dict_file_path, "w") as dict_file:
                json.dump({page_name: link}, dict_file)

    except Exception as e:
        print(f"Error saving page {link}: {e}")

def single_page_html(driver):
    """
    Scrapes the HTML content of a single web page using a Selenium WebDriver.
    Args:
        driver (WebDriver): The Selenium WebDriver instance.
    Returns:
        None
    Raises:
        None
    """
    

    url = input("Enter the URL of the page you want to see: ")
    driver.get(url)

    load_cookies(driver)

    # attend que le DOM soit stable
    if is_dom_stable(driver):
        # Find all the div widget elements
        div_widget_elements = driver.find_elements(
            By.CSS_SELECTOR, "div.widget"
        )

        print("We found", len(div_widget_elements), "widget elements")

        # Find all the span and strong elements within the widgets
        span_elements = []
        strong_elements = []
        a_elements = []
        b_elements = []
        p_elements = []
        for widget in div_widget_elements:
            span_elements.extend(widget.find_elements(By.CSS_SELECTOR, "span"))
            strong_elements.extend(
                widget.find_elements(By.CSS_SELECTOR, "strong")
            )
            a_elements.extend(widget.find_elements(By.CSS_SELECTOR, "a"))
            b_elements.extend(widget.find_elements(By.CSS_SELECTOR, "b"))
            p_elements.extend(widget.find_elements(By.CSS_SELECTOR, "p"))

        # Extract the text from the elements
        span_texts = [element.text for element in span_elements]
        strong_texts = [element.text for element in strong_elements]
        a_texts = [element.text for element in a_elements]
        b_texts = [element.text for element in b_elements]
        p_texts = [element.text for element in p_elements]

        # print the number of elements found
        print("Span elements:", len(span_elements))
        print("Strong elements:", len(strong_elements))
        print("A elements:", len(a_elements))
        print("B elements:", len(b_elements))
        print("P elements:", len(p_elements))

        print("Span texts:", span_texts)
        print("Strong texts:", strong_texts)
        print("A texts:", a_texts)
        print("B texts:", b_texts)
        print("P texts:", p_texts)

        with open("page.html", "w") as f:
            f.write(driver.page_source)

        print("Page saved as page.html")


def remove_empty_and_duplicate_html_files(directory):
    """
    Remove empty and duplicate HTML files from the given directory.

    Args:
        directory (str): The path to the directory containing the HTML files.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    """
    files = os.listdir(directory)
    html_files = [f for f in files if f.endswith(".html")]
    content_dict = {}

    empty_files_removed = 0  # Compteur pour les fichiers vides supprimés
    duplicates_removed = 0  # Compteur pour les doublons supprimés

    # Étape 1: Lire le contenu des fichiers et supprimer les vides
    for file in tqdm(html_files, desc="Processing HTML files"):
        loader = UnstructuredHTMLLoader(os.path.join(directory, file))
        data = loader.load()
        if (
            data[0].page_content.strip() == ""
            or data[0].page_content.strip() == "<div></div>"
        ):
            os.remove(os.path.join(directory, file))
            print(f"Removed empty file: {file}")
            empty_files_removed += (
                1  # Incrémenter le compteur de fichiers vides
            )
        else:
            content_dict[file] = data[0].page_content

    # Étape 2: Identifier et supprimer les doublons
    keys_to_delete = set()  # Utiliser un ensemble pour éviter les doublons
    for file1, content1 in tqdm(
        content_dict.items(), desc="Identifying duplicates"
    ):
        for file2, content2 in list(content_dict.items())[
            list(content_dict.keys()).index(file1) + 1 :
        ]:
            # print("Content1:", content1)
            # print("Content2:", content2)
            similarity = difflib.SequenceMatcher(
                None, content1, content2
            ).ratio()
            print(f"Similarity between {file1} and {file2}: {similarity}")
            if similarity >= 0.99:
                # print(f"Duplicate found: {file1} and {file2} with similarity {similarity}")
                keys_to_delete.add(
                    file2
                )  # Ajouter le fichier à supprimer dans l'ensemble

    # Supprimer les doublons et compter

    for key in keys_to_delete:
        os.remove(os.path.join(directory, key))
        duplicates_removed += 1  # Incrémenter le compteur de doublons supprimés

    # Afficher le résumé des suppressions
    print(f"Empty files removed: {empty_files_removed}")
    print(f"Duplicate files removed: {duplicates_removed}")



def main():
    parser = argparse.ArgumentParser(description="Scrape Happeo pages.")
    parser.add_argument(
        "--directory",
        type=str,
        default="data/pages",
        help="Directory to save the scraped pages (default: data/pages)"
    )
    args = parser.parse_args()
    directory = args.directory

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox") #essential !
    chrome_options.add_argument("--disable-dev-shm-usage") #essential !
    chrome_options.add_argument("--remote-debugging-port=9222") #essential !
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1200")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://app.happeo.com/home")

    no_cookies = False
    try:
        load_cookies(driver)
    except:
        print("No cookies found. Please log in manually.")
        no_cookies = True
        pass

    time.sleep(5)
    save_cookies(driver)

    visited_links = crawl_website("https://app.happeo.com/home", driver, directory)
    true_links = 0
    for i, link in tqdm(enumerate(visited_links), total=len(visited_links)):
        if link != None:
            print(f"Processing link {i+1}/{len(visited_links)}")
            print("Current link:", link)
            save_page(driver, link, directory)
            time.sleep(1)
            true_links += 1

    driver.quit()


if __name__ == "__main__":
    main()
