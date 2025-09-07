import os
import json
import time
import random
import hashlib
import logging
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# --- Config ---
KEYWORD = "Machine Learning Engineer" 
LOCATION = "United States"
OUTPUT_FILE = "dice_jobs_cyber.json"
HASH_FILE = "job_hashes_cyber.json"
MAX_JOBS = 650
TOTAL_PAGES = 100
BASE_URL = "https://www.dice.com/jobs?location={}&q={}&page="

# --- Logging Setup ---
logging.basicConfig(
    filename="logging.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

def log(msg):
    print(msg)
    logger.info(msg)

# --- Helpers ---
def make_hash(title, description):
    text = (title + description[:500]).encode("utf-8")  #first 500 chars
    return hashlib.md5(text).hexdigest()

def save_progress(job_data, seen_hashes):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2, ensure_ascii=False)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen_hashes), f, indent=2)

def load_hashes():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def is_relevant_title(title):
    t = title.lower()
    return "machine learning" in t or "ml" in t

def get_job_links(driver, keyword, page_count):
    url = BASE_URL.format(LOCATION.replace(' ', '+'), keyword.replace(' ', '+')) + str(page_count)
    driver.get(url)

    try:
        job_cards = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "div[data-testid='job-search-serp-card']")
            )
        )
    except:
        log(f"!! No job cards found on page {page_count}")
        return []

    job_links = []
    for card in job_cards:
        try:
            link = card.find_element(By.TAG_NAME, "a").get_attribute("href")
            if link and link.startswith("http"):
                job_links.append(link)
        except:
            continue

    return list(dict.fromkeys(job_links))  #unique links

def parse_job_page(driver, link, keyword):
    driver.get(link)
    time.sleep(random.uniform(2, 4))
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    title_tag = soup.find("h1", {"data-testid": "jobTitle"}) or soup.find("h1")
    desc_tag = soup.find("div", {"data-testid": "jobDescription"}) or soup.find("div", id="job-description")

    if not title_tag or not desc_tag:
        log(f"   >> Skipping (no title/description): {link}")
        return None

    title = title_tag.get_text(strip=True)
    description = desc_tag.get_text(strip=True)

    #Filter: must contain ML or Machine Learning
    if not is_relevant_title(title):
        log(f"   >> Skipping (irrelevant title): {title}")
        return None

    if len(description) < 50:
        log(f"   >> Skipping (too short description): {title}")
        return None

    log(f"   >> Scraped job: {title}")
    return {
        "title": title,
        "description": description,
        "keyword": keyword,
        "url": link
    }

# --- Main Scraper ---
def scrape_jobs():
    job_data = []
    seen_hashes = load_hashes()

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            job_data = json.load(f)

    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options)
    driver.set_window_size(1200, 800)

    # Manual login if needed
    driver.get(BASE_URL.format(LOCATION.replace(' ', '+'), KEYWORD.replace(' ', '+')) + "1")
    input(">> Log in to Dice.com manually if required, then press ENTER to start scraping...\n")

    for page_count in range(1, TOTAL_PAGES + 1):
        log(f"\n=== Scraping Page {page_count} ===")
        job_links = get_job_links(driver, KEYWORD, page_count)

        for link in job_links:
            job_entry = parse_job_page(driver, link, KEYWORD)
            if not job_entry:
                continue

            job_hash = make_hash(job_entry["title"], job_entry["description"])
            if job_hash in seen_hashes:
                log(f"   >> Skipping (duplicate): {job_entry['title']}")
                continue

            seen_hashes.add(job_hash)
            job_data.append(job_entry)

            if len(job_data) >= MAX_JOBS:
                save_progress(job_data, seen_hashes)
                log(f"\n>>> Reached {MAX_JOBS} jobs. Stopping.")
                driver.quit()
                return

        save_progress(job_data, seen_hashes)
        log(f"--- Total jobs scraped so far: {len(job_data)} ---")
        time.sleep(random.uniform(5, 8))

    driver.quit()
    save_progress(job_data, seen_hashes)
    log(f"\n>>> Finished scraping. Total jobs collected: {len(job_data)}")

if __name__ == "__main__":
    scrape_jobs()