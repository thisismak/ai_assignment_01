import os
import sqlite3
import requests
from playwright.sync_api import sync_playwright
from PIL import Image
import io
import logging
import random
from urllib.parse import urlparse

# Configure logging to track progress and errors
logging.basicConfig(
    filename='image_collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration constants
OUTPUT_DIR = "dog_images"  # Directory for downloaded images
DB_NAME = "dog_images.db"  # SQLite database name
TARGET_IMAGE_COUNT = 3000  # Minimum target (3000–5000)
MAX_IMAGE_SIZE = 50 * 1024  # 50KB max size
QUALITY_RANGE = (50, 80)  # JPEG quality range
IMAGE_SIZE = (500, 500)  # Max image dimensions
MAX_IMAGES_PER_KEYWORD = 500  # Limit per keyword to avoid timeouts

# Expanded list of dog breeds for diverse image collection
DOG_BREEDS = [
    "Labrador Retriever", "German Shepherd", "Golden Retriever",
    "Bulldog", "Poodle", "Beagle", "Rottweiler", "Siberian Husky",
    "Dachshund", "Boxer", "Shih Tzu", "Chihuahua", "Yorkshire Terrier",
    "Doberman Pinscher", "Great Dane", "Border Collie", "Australian Shepherd",
    "Cocker Spaniel", "Pug", "French Bulldog", "Maltese", "Bernese Mountain Dog",
    "Newfoundland", "Saint Bernard", "Weimaraner"
]

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def init_database():
    """Initialize SQLite database to store image metadata."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            alt_text TEXT,
            filename TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def get_keyword_variants(breed):
    """Generate keyword variants for a dog breed to improve search relevance."""
    return [
        breed,
        f"{breed} dog",
        f"{breed} puppy",
        f"cute {breed}",
        f"{breed} portrait"
    ]

def collect_image_urls_google(page, keyword, max_images):
    """Collect image URLs and alt texts from Google Images."""
    images = []
    try:
        logging.info(f"Searching Google for keyword: {keyword}")
        page.goto(f"https://www.google.com/search?tbm=isch&q={keyword}", timeout=60000)
        page.wait_for_load_state("networkidle", timeout=30000)
        last_height = page.evaluate("document.body.scrollHeight")
        while len(images) < max_images:
            image_elements = page.query_selector_all("img")
            for img in image_elements:
                src = img.get_attribute("src") or img.get_attribute("data-src")
                alt = img.get_attribute("alt") or ""
                if src and src.startswith("http") and (src, alt) not in images:
                    images.append((src, alt))
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(random.uniform(1000, 3000))  # Random delay to avoid blocking
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        logging.info(f"Collected {len(images)} images from Google for {keyword}")
    except Exception as e:
        logging.error(f"Error collecting images from Google for {keyword}: {str(e)}")
    return images

def collect_image_urls_bing(page, keyword, max_images):
    """Collect image URLs and alt texts from Bing Images."""
    images = []
    try:
        logging.info(f"Searching Bing for keyword: {keyword}")
        page.goto(f"https://www.bing.com/images/search?q={keyword}", timeout=60000)
        page.wait_for_load_state("networkidle", timeout=30000)
        last_height = page.evaluate("document.body.scrollHeight")
        while len(images) < max_images:
            image_elements = page.query_selector_all("img")
            for img in image_elements:
                src = img.get_attribute("src") or img.get_attribute("data-src")
                alt = img.get_attribute("alt") or ""
                if src and src.startswith("http") and (src, alt) not in images:
                    images.append((src, alt))
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(random.uniform(1000, 3000))  # Random delay
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        logging.info(f"Collected {len(images)} images from Bing for {keyword}")
    except Exception as e:
        logging.error(f"Error collecting images from Bing for {keyword}: {str(e)}")
    return images

def collect_image_urls(playwright, keyword, max_images):
    """Collect images from Google and Bing, deduplicating results."""
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    images = []
    try:
        images.extend(collect_image_urls_google(page, keyword, max_images))
        images.extend(collect_image_urls_bing(page, keyword, max_images))
        images = list(dict.fromkeys(images))  # Deduplicate while preserving order
        logging.info(f"Total unique images collected for {keyword}: {len(images)}")
    except Exception as e:
        logging.error(f"Error in collect_image_urls for {keyword}: {str(e)}")
    finally:
        browser.close()
    return images[:max_images]

def download_image(url, filename):
    """Download an image from a URL and save it locally."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return True
        else:
            logging.warning(f"Failed to download {url}: Status code {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def process_image(input_path, output_path):
    """Resize, center-crop, and encode image as JPEG under 50KB."""
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")  # Ensure JPEG compatibility
            img.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)
            width, height = img.size
            left = (width - min(IMAGE_SIZE[0], width)) / 2
            top = (height - min(IMAGE_SIZE[1], height)) / 2
            right = (width + min(IMAGE_SIZE[0], width)) / 2
            bottom = (height + min(IMAGE_SIZE[1], height)) / 2
            img = img.crop((left, top, right, bottom))
            quality = QUALITY_RANGE[1]
            while quality >= QUALITY_RANGE[0]:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                size = buffer.tell()
                if size <= MAX_IMAGE_SIZE:
                    with open(output_path, "wb") as f:
                        f.write(buffer.getvalue())
                    return True
                quality -= 5
            logging.warning(f"Could not compress {input_path} to under 50KB")
            return False
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    """Main function to collect, download, and process dog images."""
    conn, cursor = init_database()
    total_images = 0
    try:
        with sync_playwright() as playwright:
            for breed in DOG_BREEDS:
                if total_images >= TARGET_IMAGE_COUNT:
                    break
                for keyword in get_keyword_variants(breed):
                    if total_images >= TARGET_IMAGE_COUNT:
                        break
                    logging.info(f"Processing keyword: {keyword}")
                    images = collect_image_urls(playwright, keyword, MAX_IMAGES_PER_KEYWORD)
                    for i, (url, alt) in enumerate(images):
                        if total_images >= TARGET_IMAGE_COUNT:
                            break
                        filename = os.path.join(OUTPUT_DIR, f"{breed.replace(' ', '_')}_{total_images}.jpg")
                        temp_filename = os.path.join(OUTPUT_DIR, f"temp_{total_images}.jpg")
                        if download_image(url, temp_filename):
                            if process_image(temp_filename, filename):
                                cursor.execute(
                                    "INSERT OR IGNORE INTO images (url, alt_text, filename) VALUES (?, ?, ?)",
                                    (url, alt, filename)
                                )
                                total_images += 1
                                logging.info(f"Successfully processed image: {filename}")
                            os.remove(temp_filename) if os.path.exists(temp_filename) else None
                    conn.commit()
                    logging.info(f"Completed keyword {keyword}, total images: {total_images}")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")
    finally:
        logging.info(f"Total images collected and processed: {total_images}")
        print(f"Total images collected and processed: {total_images}")
        conn.commit()
        conn.close()

if __name__ == "__main__":
    main()