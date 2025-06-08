import os
import sqlite3
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from PIL import Image
import io
import logging
import random
import re
import csv
from urllib.parse import urlparse
from uuid import uuid4

# Configure logging to track progress and errors
logging.basicConfig(
    filename='image_collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration constants
OUTPUT_DIR = "dog_images"
DB_NAME = "dog_images.db"
TARGET_IMAGE_COUNT_MIN = 3000
TARGET_IMAGE_COUNT_MAX = 5000
IMAGES_PER_BREED = 400  # Limit per breed
MAX_IMAGE_SIZE = 100 * 1024  # 100KB to reduce compression failures
QUALITY_RANGE = (40, 90)      # Broader quality range
IMAGE_SIZE = (500, 500)
MAX_IMAGES_PER_KEYWORD = 500  # Reduced to prevent browser overload
NAVIGATION_TIMEOUT = 90000    # Increased to 90 seconds
LOAD_STATE_TIMEOUT = 45000    # Increased to 45 seconds
RETRY_ATTEMPTS = 3            # Retry attempts for failed searches

# List of dog breeds
DOG_BREEDS = [
    "Maltese", "Yorkshire Terrier", "Pomeranian", "Chihuahua", "Miniature Schnauzer",
    "Shih Tzu", "Poodle", "Dachshund", "Shiba Inu", "Labrador Retriever"
]

# Create output directory and breed-specific subdirectories
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
for breed in DOG_BREEDS:
    breed_dir = os.path.join(OUTPUT_DIR, breed.replace(' ', '_'))
    if not os.path.exists(breed_dir):
        os.makedirs(breed_dir)
        logging.info(f"Created directory for breed: {breed_dir}")

def init_database():
    """Initialize SQLite database to store image metadata."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            alt_text TEXT,
            filename TEXT,
            breed TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def get_keyword_variants(breed):
    """Generate keyword variants for a dog breed."""
    return [
        breed,
        f"{breed} dog",
        f"{breed} puppy",
        f"cute {breed}",
        f"{breed} portrait",
        f"{breed} pet",
        f"{breed} breed"
    ]

def filter_alt_text(alt):
    """Filter alt_text to ensure it's non-empty and descriptive."""
    if not alt or not alt.strip():
        logging.warning(f"Filtered out image with empty alt text")
        return False
    if len(alt.strip()) < 3:
        logging.warning(f"Filtered out image with short alt text: {alt}")
        return False
    return True

def collect_image_urls_google(page, keyword, max_images):
    """Collect image URLs and alt texts from Google Images with retries."""
    for attempt in range(RETRY_ATTEMPTS):
        images = []
        empty_alt_count = 0
        try:
            logging.info(f"Attempt {attempt+1}: Searching Google for keyword: {keyword}")
            page.goto(f"https://www.google.com/search?tbm=isch&q={keyword}", timeout=NAVIGATION_TIMEOUT)
            page.wait_for_load_state("networkidle", timeout=LOAD_STATE_TIMEOUT)
            last_height = page.evaluate("document.body.scrollHeight")
            while len(images) < max_images:
                image_elements = page.query_selector_all("img")
                for img in image_elements:
                    try:
                        src = img.get_attribute("src") or img.get_attribute("data-src")
                        alt = img.get_attribute("alt") or ""
                        if src and src.startswith("http"):
                            if filter_alt_text(alt) and (src, alt) not in images:
                                images.append((src, alt))
                            else:
                                empty_alt_count += 1
                    except Exception as e:
                        logging.warning(f"Error getting attributes for an image: {str(e)}")
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.uniform(1000, 2000))  # Increased delay
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            logging.info(f"Attempt {attempt+1}: Collected {len(images)} images from Google for {keyword}, skipped {empty_alt_count} empty/invalid alt texts")
            return images, empty_alt_count
        except PlaywrightTimeoutError as e:
            logging.error(f"Attempt {attempt+1}: Timeout in Google search for {keyword}: {str(e)}")
            if attempt < RETRY_ATTEMPTS - 1:
                logging.info(f"Retrying Google search for {keyword}")
                page.reload(timeout=NAVIGATION_TIMEOUT)
                continue
            logging.error(f"Failed Google search for {keyword} after {RETRY_ATTEMPTS} attempts")
            return [], empty_alt_count
        except Exception as e:
            logging.error(f"Attempt {attempt+1}: Error collecting images from Google for {keyword}: {str(e)}")
            if attempt < RETRY_ATTEMPTS - 1:
                logging.info(f"Retrying Google search for {keyword}")
                page.reload(timeout=NAVIGATION_TIMEOUT)
                continue
            return [], empty_alt_count
    return [], empty_alt_count

def collect_image_urls_bing(page, keyword, max_images):
    """Collect image URLs and alt texts from Bing Images with retries."""
    for attempt in range(RETRY_ATTEMPTS):
        images = []
        empty_alt_count = 0
        try:
            logging.info(f"Attempt {attempt+1}: Searching Bing for keyword: {keyword}")
            page.goto(f"https://www.bing.com/images/search?q={keyword}", timeout=NAVIGATION_TIMEOUT)
            page.wait_for_load_state("networkidle", timeout=LOAD_STATE_TIMEOUT)
            last_height = page.evaluate("document.body.scrollHeight")
            while len(images) < max_images:
                image_elements = page.query_selector_all("img")
                for img in image_elements:
                    try:
                        src = img.get_attribute("src") or img.get_attribute("data-src")
                        alt = img.get_attribute("alt") or ""
                        if src and src.startswith("http"):
                            if filter_alt_text(alt) and (src, alt) not in images:
                                images.append((src, alt))
                            else:
                                empty_alt_count += 1
                    except Exception as e:
                        logging.warning(f"Error getting attributes for an image: {str(e)}")
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.uniform(1000, 2000))  # Increased delay
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            logging.info(f"Attempt {attempt+1}: Collected {len(images)} images from Bing for {keyword}, skipped {empty_alt_count} empty/invalid alt texts")
            return images, empty_alt_count
        except PlaywrightTimeoutError as e:
            logging.error(f"Attempt {attempt+1}: Timeout in Bing search for {keyword}: {str(e)}")
            if attempt < RETRY_ATTEMPTS - 1:
                logging.info(f"Retrying Bing search for {keyword}")
                page.reload(timeout=NAVIGATION_TIMEOUT)
                continue
            logging.error(f"Failed Bing search for {keyword} after {RETRY_ATTEMPTS} attempts")
            return [], empty_alt_count
        except Exception as e:
            logging.error(f"Attempt {attempt+1}: Error collecting images from Bing for {keyword}: {str(e)}")
            if attempt < RETRY_ATTEMPTS - 1:
                logging.info(f"Retrying Bing search for {keyword}")
                page.reload(timeout=NAVIGATION_TIMEOUT)
                continue
            return [], empty_alt_count
    return [], empty_alt_count

def collect_image_urls(playwright, keyword, max_images):
    """Collect images from Google and Bing, deduplicating results."""
    images = []
    total_empty_alt_count = 0
    browser = None
    try:
        browser = playwright.chromium.launch(headless=False)
        # NEW: Use browser context for better resource management
        context = browser.new_context()
        page = context.new_page()
        google_images, google_empty_alt = collect_image_urls_google(page, keyword, int(max_images * 2))
        images.extend(google_images)
        total_empty_alt_count += google_empty_alt
        bing_images, bing_empty_alt = collect_image_urls_bing(page, keyword, int(max_images * 2))
        images.extend(bing_images)
        total_empty_alt_count += bing_empty_alt
        images = list(dict.fromkeys(images))  # Deduplicate
        logging.info(f"Total unique images collected for {keyword}: {len(images)}, total skipped alt texts: {total_empty_alt_count}")
        return images[:max_images], total_empty_alt_count
    except Exception as e:
        logging.error(f"Error in collect_image_urls for {keyword}: {str(e)}")
        return [], total_empty_alt_count
    finally:
        if browser:
            try:
                context.close()
                browser.close()
            except Exception as e:
                logging.error(f"Error closing browser: {str(e)}")

def download_image(url, filename):
    """Download an image from a URL and save it locally with retries."""
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                return True
            else:
                logging.warning(f"Attempt {attempt+1}: Failed to download {url}: Status code {response.status_code}")
        except Exception as e:
            logging.error(f"Attempt {attempt+1}: Error downloading {url}: {str(e)}")
    logging.error(f"Failed to download {url} after 3 attempts")
    return False

def process_image(input_path, output_path):
    """Resize, center-crop, and encode image as JPEG under MAX_IMAGE_SIZE."""
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
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
            logging.warning(f"Could not compress {input_path} to under {MAX_IMAGE_SIZE/1024}KB")
            return False
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        return False

def export_failure_report(failure_log, filename="image_failure_report.csv"):
    """Export failure log to CSV for analysis."""
    headers = ["keyword", "url", "alt_text", "failure_stage", "failure_reason"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(failure_log)
    logging.info(f"Exported failure report to {filename}")

def export_breed_distribution(breed_counts, filename="breed_distribution.csv"):
    """Export breed distribution to CSV for analysis."""
    headers = ["breed", "image_count"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for breed, count in breed_counts.items():
            writer.writerow([breed, count])
    logging.info(f"Exported breed distribution to {filename}")

def main():
    """Main function to collect, download, and process dog images."""
    conn, cursor = init_database()
    total_images = 0
    filtered_out_alt = 0
    download_failed = 0
    process_failed = 0
    duplicate_urls = 0
    failure_log = []
    breed_counts = {breed: 0 for breed in DOG_BREEDS}  # Track images per breed

    try:
        # Randomize breed order to avoid bias
        randomized_breeds = DOG_BREEDS.copy()
        random.shuffle(randomized_breeds)
        
        with sync_playwright() as playwright:
            # First pass: collect up to IMAGES_PER_BREED per breed
            for breed in randomized_breeds:
                if total_images >= TARGET_IMAGE_COUNT_MAX:
                    break
                for keyword in get_keyword_variants(breed):
                    if total_images >= TARGET_IMAGE_COUNT_MAX or breed_counts[breed] >= IMAGES_PER_BREED:
                        break
                    logging.info(f"Processing keyword: {keyword} for breed: {breed}")
                    images, empty_alt_count = collect_image_urls(playwright, keyword, MAX_IMAGES_PER_KEYWORD)
                    filtered_out_alt += empty_alt_count
                    for i, (url, alt) in enumerate(images):
                        if total_images >= TARGET_IMAGE_COUNT_MAX or breed_counts[breed] >= IMAGES_PER_BREED:
                            break
                        breed_dir = os.path.join(OUTPUT_DIR, breed.replace(' ', '_'))
                        filename = os.path.join(breed_dir, f"{breed.replace(' ', '_')}_{breed_counts[breed]}_{total_images}.jpg")
                        temp_filename = os.path.join(breed_dir, f"temp_{breed_counts[breed]}_{total_images}.jpg")
                        if not download_image(url, temp_filename):
                            download_failed += 1
                            failure_log.append([keyword, url, alt, "download", "Failed after 3 attempts"])
                            continue
                        if not process_image(temp_filename, filename):
                            process_failed += 1
                            failure_log.append([keyword, url, alt, "processing", "Compression or processing error"])
                            os.remove(temp_filename) if os.path.exists(temp_filename) else None
                            continue
                        cursor.execute(
                            "INSERT OR IGNORE INTO images (url, alt_text, filename, breed) VALUES (?, ?, ?, ?)",
                            (url, alt, filename, breed)
                        )
                        if cursor.rowcount == 0:
                            duplicate_urls += 1
                            failure_log.append([keyword, url, alt, "database", "Duplicate URL"])
                        else:
                            total_images += 1
                            breed_counts[breed] += 1
                            logging.info(f"Successfully processed image: {filename} for breed: {breed}")
                        os.remove(temp_filename) if os.path.exists(temp_filename) else None
                    conn.commit()
                    logging.info(f"Completed keyword {keyword}, total images: {total_images}, breed {breed}: {breed_counts[breed]}")
            
            # Second pass: supplement if total_images < TARGET_IMAGE_COUNT_MIN
            if total_images < TARGET_IMAGE_COUNT_MIN:
                logging.info(f"Shortfall of {TARGET_IMAGE_COUNT_MIN - total_images} images, retrying underfilled breeds")
                underfilled_breeds = [breed for breed, count in breed_counts.items() if count < IMAGES_PER_BREED]
                random.shuffle(underfilled_breeds)
                for breed in underfilled_breeds:
                    if total_images >= TARGET_IMAGE_COUNT_MIN:
                        break
                    for keyword in get_keyword_variants(breed):
                        if total_images >= TARGET_IMAGE_COUNT_MIN or breed_counts[breed] >= IMAGES_PER_BREED:
                            break
                        logging.info(f"Supplementing keyword {keyword} for breed {breed}")
                        images, empty_alt_count = collect_image_urls(playwright, keyword, MAX_IMAGES_PER_KEYWORD)
                        filtered_out_alt += empty_alt_count
                        for i, (url, alt) in enumerate(images):
                            if total_images >= TARGET_IMAGE_COUNT_MIN or breed_counts[breed] >= IMAGES_PER_BREED:
                                break
                            breed_dir = os.path.join(OUTPUT_DIR, breed.replace(' ', '_'))
                            filename = os.path.join(breed_dir, f"{breed.replace(' ', '_')}_{breed_counts[breed]}_{total_images}.jpg")
                            temp_filename = os.path.join(breed_dir, f"temp_{breed_counts[breed]}_{total_images}.jpg")
                            if not download_image(url, temp_filename):
                                download_failed += 1
                                failure_log.append([keyword, url, alt, "download", "Failed after 3 attempts"])
                                continue
                            if not process_image(temp_filename, filename):
                                process_failed += 1
                                failure_log.append([keyword, url, alt, "processing", "Compression or processing error"])
                                os.remove(temp_filename) if os.path.exists(temp_filename) else None
                                continue
                            cursor.execute(
                                "INSERT OR IGNORE INTO images (url, alt_text, filename, breed) VALUES (?, ?, ?, ?)",
                                (url, alt, filename, breed)
                            )
                            if cursor.rowcount == 0:
                                duplicate_urls += 1
                                failure_log.append([keyword, url, alt, "database", "Duplicate URL"])
                            else:
                                total_images += 1
                                breed_counts[breed] += 1
                                logging.info(f"Successfully processed image: {filename} for breed: {breed}")
                            os.remove(temp_filename) if os.path.exists(temp_filename) else None
                        conn.commit()
                        logging.info(f"Completed supplementing keyword {keyword}, total images: {total_images}, breed {breed}: {breed_counts[breed]}")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")
        failure_log.append(["N/A", "N/A", "N/A", "main_loop", str(e)])
    finally:
        # Verify database count
        cursor.execute("SELECT COUNT(*) FROM images")
        db_count = cursor.fetchone()[0]
        logging.info(f"Database verification: {db_count} images stored")
        print(f"數據庫驗證：存儲了 {db_count} 張圖片")
        
        # Log summary
        logging.info(f"總共收集並處理的圖片數：{total_images}")
        logging.info(f"因無效alt文本過濾掉的圖片：{filtered_out_alt}")
        logging.info(f"下載失敗的圖片：{download_failed}")
        logging.info(f"處理失敗的圖片：{process_failed}")
        logging.info(f"重複URL跳過的圖片：{duplicate_urls}")
        logging.info(f"品種分佈：{breed_counts}")
        print(f"總共收集並處理的圖片數：{total_images}")
        print(f"因無效alt文本過濾掉的圖片：{filtered_out_alt}")
        print(f"下載失敗的圖片：{download_failed}")
        print(f"處理失敗的圖片：{process_failed}")
        print(f"重複URL跳過的圖片：{duplicate_urls}")
        print(f"品種分佈：{breed_counts}")
        
        # Export reports
        export_failure_report(failure_log)
        export_breed_distribution(breed_counts)
        
        conn.commit()
        conn.close()

if __name__ == "__main__":
    main()