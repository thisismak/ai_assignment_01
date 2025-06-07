import os
import sqlite3
import logging
import csv
from urllib.parse import urlparse
from PIL import Image
import imagehash
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from collections import defaultdict
import random
import requests
from uuid import uuid4
import re

# Configure logging for tracking progress and errors
logging.basicConfig(
    filename='dataset_cleaning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration constants
OUTPUT_DIR = "dog_images"
DB_NAME = "dog_images.db"
TARGET_IMAGE_COUNT_MIN = 1000
TARGET_IMAGE_COUNT_MAX = 2000
CLEANED_DIR = "cleaned_dog_images"
TEMP_DIR = "temp_images"
MODEL_CONFIDENCE_THRESHOLD = 0.9  # Confidence threshold for dog classification
HASH_SIZE = 8  # Size for perceptual hash
PAGE_COUNT_LOG = "image_collection.log"  # Log file from Exercise 1

# Initialize ResNet50 model for dog classification
model = ResNet50(weights='imagenet')

class DatasetCleaner:
    """Class to handle image dataset cleaning and analysis."""
    
    def __init__(self):
        """Initialize database connection, create images table, and output directories."""
        self.conn = sqlite3.connect(DB_NAME)
        self.cursor = self.conn.cursor()
        # Ensure images table exists
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                alt_text TEXT,
                filename TEXT,
                breed TEXT
            )
        ''')
        self.conn.commit()
        if not os.path.exists(CLEANED_DIR):
            os.makedirs(CLEANED_DIR)
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        self.image_hashes = {}
        self.failure_log = []
        self.breed_counts = defaultdict(int)
        self.unique_domains = set()
        self.page_count = 0

    def is_dog_image(self, image_path):
        """Classify if an image contains a dog using ResNet50."""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            predictions = model.predict(img_array, verbose=0)
            predicted_class = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=1)[0][0]
            class_id, class_name, confidence = predicted_class
            # Check if the predicted class is a dog (ImageNet classes 151-268 are dogs)
            is_dog = 151 <= int(class_id.split('_')[1]) <= 268 if class_id.startswith('n') else False
            logging.info(f"Image {image_path}: Classified as {class_name} with confidence {confidence}, is_dog: {is_dog}")
            return is_dog and confidence >= MODEL_CONFIDENCE_THRESHOLD
        except Exception as e:
            logging.error(f"Error classifying {image_path}: {str(e)}")
            return False

    def compute_image_hash(self, image_path):
        """Compute perceptual hash of an image."""
        try:
            with Image.open(image_path) as img:
                return str(imagehash.average_hash(img, hash_size=HASH_SIZE))
        except Exception as e:
            logging.error(f"Error computing hash for {image_path}: {str(e)}")
            return None

    def check_database(self):
        """Check if the images table contains data."""
        self.cursor.execute("SELECT COUNT(*) FROM images")
        count = self.cursor.fetchone()[0]
        if count == 0:
            logging.error("No data found in images table. Please run script.py first.")
            return False
        return True

    def clean_dataset(self):
        """Clean the dataset by removing irrelevant and duplicate images."""
        if not self.check_database():
            logging.error("Cannot proceed with cleaning due to empty database.")
            return 0, 0, 0

        try:
            self.cursor.execute("SELECT id, url, alt_text, filename, breed FROM images")
            images = self.cursor.fetchall()
        except sqlite3.OperationalError as e:
            logging.error(f"Database error: {str(e)}. Ensure images table exists.")
            return 0, 0, 0

        cleaned_count = 0
        irrelevant_count = 0
        duplicate_count = 0

        for img_id, url, alt_text, filename, breed in images:
            if not os.path.exists(filename):
                logging.warning(f"Image file {filename} not found")
                self.failure_log.append([breed, url, alt_text, "file_missing", "File not found"])
                self.cursor.execute("DELETE FROM images WHERE id = ?", (img_id,))
                continue

            # Check if image is relevant (contains a dog)
            if not self.is_dog_image(filename):
                irrelevant_count += 1
                self.failure_log.append([breed, url, alt_text, "classification", "Not a dog image"])
                os.remove(filename) if os.path.exists(filename) else None
                self.cursor.execute("DELETE FROM images WHERE id = ?", (img_id,))
                continue

            # Check for duplicates using perceptual hash
            img_hash = self.compute_image_hash(filename)
            if img_hash is None:
                self.failure_log.append([breed, url, alt_text, "hashing", "Failed to compute hash"])
                os.remove(filename) if os.path.exists(filename) else None
                self.cursor.execute("DELETE FROM images WHERE id = ?", (img_id,))
                continue
            if img_hash in self.image_hashes:
                duplicate_count += 1
                self.failure_log.append([breed, url, alt_text, "duplicate", f"Duplicate of {self.image_hashes[img_hash]}"])
                os.remove(filename) if os.path.exists(filename) else None
                self.cursor.execute("DELETE FROM images WHERE id = ?", (img_id,))
                continue
            self.image_hashes[img_hash] = filename

            # Move to cleaned directory
            cleaned_filename = os.path.join(CLEANED_DIR, os.path.basename(filename))
            try:
                os.rename(filename, cleaned_filename)
                self.cursor.execute("UPDATE images SET filename = ? WHERE id = ?", (cleaned_filename, img_id))
            except OSError as e:
                logging.error(f"Error moving file {filename} to {cleaned_filename}: {str(e)}")
                self.failure_log.append([breed, url, alt_text, "file_move", str(e)])
                self.cursor.execute("DELETE FROM images WHERE id = ?", (img_id,))
                continue

            self.breed_counts[breed] += 1
            cleaned_count += 1

            # Extract domain for source analysis
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            self.unique_domains.add(domain)

        self.conn.commit()
        logging.info(f"Cleaned dataset: {cleaned_count} images kept, {irrelevant_count} irrelevant, {duplicate_count} duplicates")
        return cleaned_count, irrelevant_count, duplicate_count

    def supplement_images(self, target_min, target_max, original_script):
        """Supplement dataset if cleaned count is below target minimum."""
        cleaned_count, _, _ = self.clean_dataset()
        if cleaned_count >= target_min:
            return cleaned_count

        logging.info(f"Supplementing dataset: {target_min - cleaned_count} images needed")
        try:
            from script import main as collect_images
            collect_images()  # Run original script to collect more images
            self.conn.commit()
            return self.clean_dataset()[0]  # Re-clean after supplementing
        except ImportError as e:
            logging.error(f"Failed to import original script: {str(e)}")
            self.failure_log.append(["N/A", "N/A", "N/A", "supplement", "Failed to import original script"])
            return cleaned_count
        except Exception as e:
            logging.error(f"Error supplementing images: {str(e)}")
            self.failure_log.append(["N/A", "N/A", "N/A", "supplement", str(e)])
            return cleaned_count

    def count_pages(self):
        """Count the number of pages crawled from the log file."""
        try:
            if os.path.exists(PAGE_COUNT_LOG):
                with open(PAGE_COUNT_LOG, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    self.page_count = len(re.findall(r'Searching (Google|Bing) for keyword', log_content))
                logging.info(f"Total pages crawled: {self.page_count}")
            else:
                logging.warning(f"Log file {PAGE_COUNT_LOG} not found")
                self.page_count = 0
        except Exception as e:
            logging.error(f"Error counting pages: {str(e)}")
            self.page_count = 0

    def generate_report(self, cleaned_count, irrelevant_count, duplicate_count):
        """Generate CSV report with statistics and source analysis."""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM images")
            original_count = self.cursor.fetchone()[0]
        except sqlite3.OperationalError as e:
            logging.error(f"Error querying database for report: {str(e)}")
            original_count = 0

        report_data = [
            ["Metric", "Value"],
            ["Original Image Count", original_count],
            ["Cleaned Image Count", cleaned_count],
            ["Irrelevant Images Removed", irrelevant_count],
            ["Duplicate Images Removed", duplicate_count],
            ["Unique Domains", len(self.unique_domains)],
            ["Pages Crawled", self.page_count]
        ]
        with open("dataset_report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(report_data)
        logging.info("Generated dataset report")

        # Export breed distribution
        with open("cleaned_breed_distribution.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Breed", "Image Count"])
            for breed, count in self.breed_counts.items():
                writer.writerow([breed, count])
        logging.info("Generated breed distribution report")

        # Export failure log
        with open("cleaning_failure_report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Breed", "URL", "Alt Text", "Failure Stage", "Failure Reason"])
            writer.writerows(self.failure_log)
        logging.info("Generated failure report")

    def close(self):
        """Close database connection."""
        try:
            self.conn.commit()
            self.conn.close()
        except Exception as e:
            logging.error(f"Error closing database connection: {str(e)}")

def main():
    """Main function to clean dataset and generate reports."""
    cleaner = DatasetCleaner()
    try:
        cleaner.count_pages()
        cleaned_count, irrelevant_count, duplicate_count = cleaner.clean_dataset()
        if cleaned_count == 0:
            print("清理失敗：數據庫為空或無有效圖像數據。請先運行 script.py 收集圖像。")
            logging.error("Cleaning aborted due to empty dataset or database error.")
            return
        if cleaned_count < TARGET_IMAGE_COUNT_MIN:
            cleaned_count = cleaner.supplement_images(TARGET_IMAGE_COUNT_MIN, TARGET_IMAGE_COUNT_MAX, None)
        cleaner.generate_report(cleaned_count, irrelevant_count, duplicate_count)
        logging.info("Dataset cleaning and reporting completed")
        print(f"數據集清理完成：")
        print(f"原始圖像數量：{cleaner.cursor.execute('SELECT COUNT(*) FROM images').fetchone()[0]}")
        print(f"清理後圖像數量：{cleaned_count}")
        print(f"移除的不相關圖像：{irrelevant_count}")
        print(f"移除的重複圖像：{duplicate_count}")
        print(f"唯一網域數量：{len(cleaner.unique_domains)}")
        print(f"爬取的頁數：{cleaner.page_count}")
        print(f"品種分佈：{dict(cleaner.breed_counts)}")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(f"程式執行失敗：{str(e)}")
    finally:
        cleaner.close()

if __name__ == "__main__":
    main()