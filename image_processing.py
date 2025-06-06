import os
import sqlite3
import requests
import logging
from logging.handlers import RotatingFileHandler
import csv
import random
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Configure logging with rotation to limit file size
handler = RotatingFileHandler('image_processing.log', maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,  # Changed to INFO for detailed debugging
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define constants
OUTPUT_DIR = "dog_images"
TRAINING_DATA_DIR = "training_data"
DB_NAME = "dog_images.db"
TARGET_IMAGE_COUNT_MIN = 1000
TARGET_IMAGE_COUNT_MAX = 5000
IMAGES_PER_BREED = 400
MAX_IMAGE_SIZE = 200 * 1024  # Increased to 200KB
QUALITY_RANGE = (40, 90)
IMAGE_SIZE = (500, 500)
CLASSIFIER_IMAGE_SIZE = (224, 224)
MAX_IMAGES_PER_KEYWORD = 1000
TRAINING_IMAGES_PER_CLASS = 500  # Increased to 500 images

# List of dog breeds
DOG_BREEDS = [
    "Maltese", "Yorkshire Terrier", "Pomeranian", "Chihuahua", "Miniature Schnauzer",
    "Shih Tzu", "Poodle", "Dachshund", "Shiba Inu", "Labrador Retriever"
]

# Create output and training data directories
for directory in [OUTPUT_DIR, os.path.join(TRAINING_DATA_DIR, "real_dogs"), os.path.join(TRAINING_DATA_DIR, "non_real_dogs")]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

class DatabaseManager:
    """Manage SQLite database operations"""
    def __init__(self, db_name):
        """Initialize database connection"""
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.init_database()
    
    def init_database(self):
        """Initialize database and create images table"""
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
    
    def get_all_images(self):
        """Retrieve all image records"""
        self.cursor.execute("SELECT id, url, alt_text, filename, breed FROM images")
        return self.cursor.fetchall()
    
    def delete_image(self, image_id):
        """Delete an image record from the database"""
        self.cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        self.conn.commit()
    
    def get_image_count(self):
        """Get total number of images in the database"""
        self.cursor.execute("SELECT COUNT(*) FROM images")
        return self.cursor.fetchone()[0]
    
    def get_unique_domains(self):
        """Get unique domains of image sources"""
        self.cursor.execute("SELECT url FROM images")
        urls = [row[0] for row in self.cursor.fetchall()]
        domains = {urlparse(url).netloc for url in urls}
        return domains
    
    def get_page_count(self):
        """Estimate crawled pages based on URL count"""
        self.cursor.execute("SELECT COUNT(DISTINCT url) FROM images")
        return self.cursor.fetchone()[0] // 50 + 1
    
    def close(self):
        """Close database connection"""
        self.conn.commit()
        self.conn.close()

class ImageCollector:
    """Responsible for collecting and downloading images from Google and Bing"""
    def __init__(self, output_dir, db_manager):
        """Initialize image collector"""
        self.output_dir = output_dir
        self.db_manager = db_manager
        self.failure_log = []
    
    def get_keyword_variants(self, breed):
        """Generate keyword variants for dog breeds, prioritizing real dog photos"""
        return [
            f"{breed} real dog photo",
            f"{breed} dog photograph",
            f"{breed} puppy photo",
            f"real {breed} portrait",
            f"{breed} pet photo",
            f"{breed} breed photo"
        ]
    
    def filter_alt_text(self, alt, is_non_real=False):
        """Filter invalid or irrelevant alt text, excluding cartoons and tattoos for real dogs"""
        if not alt or not alt.strip():
            logging.warning(f"Filtered out image with empty alt text")
            return False
        if len(alt.strip()) < 3:
            logging.warning(f"Filtered out image with short alt text: {alt}")
            return False
        if not is_non_real:  # Apply blacklist only for real dog images
            blacklist = ["cartoon", "tattoo"]
            if any(keyword in alt.lower() for keyword in blacklist):
                logging.warning(f"Filtered out image with blacklisted alt text: {alt}")
                return False
        return True
    
    def collect_image_urls_google(self, page, keyword, max_images, is_non_real=False):
        """Collect image URLs and alt text from Google Images"""
        images = []
        empty_alt_count = 0
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
                    if src and src.startswith("http"):
                        if self.filter_alt_text(alt, is_non_real) and (src, alt) not in images:
                            images.append((src, alt))
                        else:
                            empty_alt_count += 1
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.uniform(500, 1500))
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            logging.info(f"Collected {len(images)} images from Google for {keyword}, skipped {empty_alt_count} empty/invalid alt texts")
        except Exception as e:
            logging.error(f"Error collecting images from Google for {keyword}: {str(e)}")
        return images, empty_alt_count
    
    def collect_image_urls_bing(self, page, keyword, max_images, is_non_real=False):
        """Collect image URLs and alt text from Bing Images"""
        images = []
        empty_alt_count = 0
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
                    if src and src.startswith("http"):
                        if self.filter_alt_text(alt, is_non_real) and (src, alt) not in images:
                            images.append((src, alt))
                        else:
                            empty_alt_count += 1
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.uniform(500, 1500))
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            logging.info(f"Collected {len(images)} images from Bing for {keyword}, skipped {empty_alt_count} empty/invalid alt texts")
        except Exception as e:
            logging.error(f"Error collecting images from Bing for {keyword}: {str(e)}")
        return images, empty_alt_count
    
    def collect_image_urls(self, keyword, max_images, is_non_real=False):
        """Collect images from Google and Bing, deduplicate and return"""
        images = []
        total_empty_alt_count = 0
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124")
            try:
                google_images, google_empty_alt = self.collect_image_urls_google(page, keyword, int(max_images * 2), is_non_real)
                images.extend(google_images)
                total_empty_alt_count += google_empty_alt
                bing_images, bing_empty_alt = self.collect_image_urls_bing(page, keyword, int(max_images * 2), is_non_real)
                images.extend(bing_images)
                total_empty_alt_count += bing_empty_alt
                images = list(dict.fromkeys(images))
                logging.info(f"Total unique images collected for {keyword}: {len(images)}, total skipped alt texts: {total_empty_alt_count}")
            except Exception as e:
                logging.error(f"Error in collect_image_urls for {keyword}: {str(e)}")
            finally:
                browser.close()
        return images[:max_images], total_empty_alt_count
    
    def download_image(self, url, filename):
        """Download image and save to local"""
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    img_data = response.content
                    try:
                        Image.open(io.BytesIO(img_data)).verify()
                    except Exception as e:
                        logging.error(f"Invalid image file from {url}: {str(e)}")
                        return False
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    return True
                else:
                    logging.warning(f"Attempt {attempt+1}: Failed to download {url}: Status code {response.status_code}")
            except Exception as e:
                logging.error(f"Attempt {attempt+1}: Error downloading {url}: {str(e)}")
        logging.error(f"Failed to download {url} after 3 attempts")
        return False
    
    def process_image(self, input_path, output_path):
        """Resize, crop, and compress image to JPEG"""
        try:
            with Image.open(input_path) as img:
                if img.format not in ['JPEG', 'PNG', 'WEBP']:
                    logging.error(f"Unsupported format for {input_path}: {img.format}")
                    return False
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
    
    def collect_training_data(self, keyword, max_images, is_non_real=False):
        """Collect training data and save to specified directory"""
        images, empty_alt_count = self.collect_image_urls(keyword, max_images, is_non_real)
        for i, (url, alt) in enumerate(images):
            filename = os.path.join(self.output_dir, f"{keyword.replace(' ', '_')}_{i}.jpg")
            if not self.download_image(url, filename):
                self.failure_log.append([keyword, url, alt, "download", "Failed after 3 attempts"])
                continue
            if not self.process_image(filename, filename):
                self.failure_log.append([keyword, url, alt, "processing", "Compression or processing error"])
                if os.path.exists(filename):
                    os.remove(filename)
                continue
            logging.info(f"Collected and processed training image: {filename} for keyword: {keyword}")
    
    def collect_and_process(self):
        """Collect and process images, store in database"""
        total_images = 0
        filtered_out_alt = 0
        download_failed = 0
        process_failed = 0
        duplicate_urls = 0
        breed_counts = {breed: 0 for breed in DOG_BREEDS}
        
        randomized_breeds = DOG_BREEDS.copy()
        random.shuffle(randomized_breeds)
        
        for breed in randomized_breeds:
            if total_images >= TARGET_IMAGE_COUNT_MAX:
                break
            for keyword in self.get_keyword_variants(breed):
                if total_images >= TARGET_IMAGE_COUNT_MAX or breed_counts[breed] >= IMAGES_PER_BREED:
                    break
                logging.info(f"Processing keyword: {keyword} for breed: {breed}")
                images, empty_alt_count = self.collect_image_urls(keyword, MAX_IMAGES_PER_KEYWORD)
                filtered_out_alt += empty_alt_count
                for i, (url, alt) in enumerate(images):
                    if total_images >= TARGET_IMAGE_COUNT_MAX or breed_counts[breed] >= IMAGES_PER_BREED:
                        break
                    filename = os.path.join(self.output_dir, f"{breed.replace(' ', '_')}_{total_images}.jpg")
                    temp_filename = os.path.join(self.output_dir, f"temp_{total_images}.jpg")
                    try:
                        if not self.download_image(url, temp_filename):
                            download_failed += 1
                            self.failure_log.append([keyword, url, alt, "download", "Failed after 3 attempts"])
                            continue
                        if not self.process_image(temp_filename, filename):
                            process_failed += 1
                            self.failure_log.append([keyword, url, alt, "processing", "Compression or processing error"])
                            continue
                        self.db_manager.cursor.execute(
                            "INSERT INTO images (url, alt_text, filename, breed) VALUES (?, ?, ?, ?)",
                            (url, alt, filename, breed)
                        )
                        self.db_manager.conn.commit()
                        total_images += 1
                        breed_counts[breed] += 1
                        logging.info(f"Successfully processed and inserted image: {filename} for breed: {breed}")
                    except sqlite3.Error as e:
                        duplicate_urls += 1
                        self.failure_log.append([keyword, url, alt, "database", f"Database error: {str(e)}"])
                    finally:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                            logging.info(f"Removed temporary file: {temp_filename}")
                logging.info(f"Completed keyword {keyword}, total images: {total_images}, breed {breed}: {breed_counts[breed]}")
        logging.info(f"Total images in database after collection: {self.db_manager.get_image_count()}")
        return total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts

class ImageClassifier:
    """Image classifier using fine-tuned MobileNetV2 to identify real dog photos"""
    def __init__(self, train_data_dir=None):
        """Initialize and fine-tune MobileNetV2 model"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=x)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        if train_data_dir:
            train_generator, val_generator = self.load_training_data(train_data_dir)
            history = self.model.fit(
                train_generator,
                epochs=5,
                validation_data=val_generator,
                verbose=1
            )
            self.plot_training_history(history, "training_history.png")
            self.model.save("real_dog_classifier.h5")
            logging.info(f"Training completed: final training accuracy {history.history['accuracy'][-1]:.4f}, "
                        f"validation accuracy {history.history['val_accuracy'][-1]:.4f}")
    
    def load_training_data(self, data_dir, target_size=(224, 224), batch_size=32):
        """Load training data"""
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        val_generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
        return train_generator, val_generator
    
    def plot_training_history(self, history, output_file):
        """Plot training loss and accuracy curves"""
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Training history plot saved to {output_file}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for MobileNetV2"""
        try:
            img = load_img(image_path, target_size=CLASSIFIER_IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            return img_array
        except Exception as e:
            logging.error(f"Image preprocessing failed for {image_path}: {str(e)}")
            return None
    
    def is_real_dog_image(self, image_path):
        """Determine if an image is a real dog photo"""
        try:
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return False
            pred = self.model.predict(img_array)[0][0]
            logging.info(f"Prediction for {image_path}: {pred:.4f}")  # Log prediction score
            if 0.4 <= pred < 0.5:
                logging.warning(f"Low confidence real dog image: {image_path}, prob: {pred:.4f}")
            return pred > 0.5  # Adjusted threshold
        except Exception as e:
            logging.error(f"Image classification failed for {image_path}: {str(e)}")
            return False

class DatasetCleaner:
    """Responsible for cleaning dataset and generating reports"""
    def __init__(self, output_dir, db_manager, classifier):
        """Initialize dataset cleaner"""
        self.output_dir = output_dir
        self.db_manager = db_manager
        self.classifier = classifier
        self.failure_log = []
    
    def detect_duplicates(self, image_files):
        """Detect duplicate images using simple hash comparison"""
        hashes = {}
        duplicates = []
        for filename in image_files:
            if not os.path.exists(filename):
                logging.error(f"File not found: {filename}")
                self.failure_log.append(["N/A", filename, "N/A", "duplicate_detection", "File not found"])
                continue
            try:
                with Image.open(filename) as img:
                    img = img.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
                    pixels = list(img.getdata())
                    avg_pixel = sum(pixels) / len(pixels)
                    hash_val = ''.join('1' if p > avg_pixel else '0' for p in pixels)
                    if hash_val in hashes:
                        duplicates.append(filename)
                    else:
                        hashes[hash_val] = filename
            except Exception as e:
                logging.error(f"Duplicate detection failed for {filename}: {str(e)}")
                self.failure_log.append(["N/A", filename, "N/A", "duplicate_detection", str(e)])
        return duplicates
    
    def clean_dataset(self):
        """Clean dataset by removing irrelevant and duplicate images"""
        images = self.db_manager.get_all_images()
        initial_count = len(images)
        removed_count = 0
        cartoon_tattoo_count = 0
        pred_scores = []  # Track prediction scores
        image_files = [os.path.join(self.output_dir, row[3]) for row in images]
        logging.info(f"Checking files: {image_files[:5]}")
        
        duplicates = self.detect_duplicates(image_files)
        for dup in duplicates:
            image_id = next((row[0] for row in images if os.path.join(self.output_dir, row[3]) == dup), None)
            if image_id:
                self.db_manager.delete_image(image_id)
                if os.path.exists(dup):
                    os.remove(dup)
                removed_count += 1
                self.failure_log.append(["N/A", dup, "N/A", "duplicate_removal", "Duplicate image"])
        
        for image_id, url, alt_text, filename, breed in images:
            file_path = os.path.join(self.output_dir, filename)
            if not os.path.exists(file_path):
                self.db_manager.delete_image(image_id)
                removed_count += 1
                self.failure_log.append([breed, url, alt_text, "file_check", "File not found"])
                continue
            img_array = self.classifier.preprocess_image(file_path)
            if img_array is None:
                self.db_manager.delete_image(image_id)
                if os.path.exists(file_path):
                    os.remove(file_path)
                removed_count += 1
                self.failure_log.append([breed, url, alt_text, "classification", "Preprocessing failed"])
                continue
            pred = self.classifier.model.predict(img_array)[0][0]
            pred_scores.append(pred)
            if not self.classifier.is_real_dog_image(file_path):
                self.db_manager.delete_image(image_id)
                if os.path.exists(file_path):
                    os.remove(file_path)
                removed_count += 1
                reason = "Not a real dog image"
                if "cartoon" in alt_text.lower() or "tattoo" in alt_text.lower():
                    reason = "Cartoon or tattoo image"
                    cartoon_tattoo_count += 1
                self.failure_log.append([breed, url, alt_text, "classification", reason])
        
        # Log prediction score statistics
        if pred_scores:
            mean_pred = np.mean(pred_scores)
            std_pred = np.std(pred_scores)
            logging.info(f"Prediction score stats: mean={mean_pred:.4f}, std={std_pred:.4f}, min={min(pred_scores):.4f}, max={max(pred_scores):.4f}")
        
        final_count = self.db_manager.get_image_count()
        logging.info(f"Initial image count: {initial_count}, Removed images: {removed_count}, Cartoon/tattoo images: {cartoon_tattoo_count}, Final image count: {final_count}")
        print(f"Data report: Cartoon/tattoo images removed: {cartoon_tattoo_count}")
        return initial_count, removed_count, final_count, cartoon_tattoo_count
    
    def supplement_dataset(self, collector):
        """Supplement dataset to reach TARGET_IMAGE_COUNT_MIN"""
        final_count = self.db_manager.get_image_count()
        if final_count < TARGET_IMAGE_COUNT_MIN:
            logging.info(f"Image count {final_count} is below minimum {TARGET_IMAGE_COUNT_MIN}, supplementing")
            total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts = collector.collect_and_process()
            return total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts
        return None, None, None, None, None, None
    
    def generate_report(self):
        """Generate data report"""
        initial_count, removed_count, final_count, cartoon_tattoo_count = self.clean_dataset()
        unique_domains = self.db_manager.get_unique_domains()
        page_count = self.db_manager.get_page_count()
        breed_counts = {breed: 0 for breed in DOG_BREEDS}
        for breed in DOG_BREEDS:
            self.db_manager.cursor.execute("SELECT COUNT(*) FROM images WHERE breed = ?", (breed,))
            breed_counts[breed] = self.db_manager.cursor.fetchone()[0]
        
        report = {
            "Initial image count": initial_count,
            "Removed images": removed_count,
            "Cartoon/tattoo images removed": cartoon_tattoo_count,
            "Final image count": final_count,
            "Crawled pages": page_count,
            "Unique domains": len(unique_domains),
            "Breed distribution": breed_counts
        }
        logging.info(f"Data report: {report}")
        print(f"Data report: {report}")
        
        self.export_failure_report()
        self.export_breed_distribution(breed_counts)
        
        return report, breed_counts
    
    def export_failure_report(self, filename="image_processing_failure_report.csv"):
        """Export failure report"""
        headers = ["keyword/breed", "url", "alt_text", "failure_stage", "failure_reason"]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.failure_log)
            if not self.failure_log:
                writer.writerow(["N/A", "N/A", "N/A", "N/A", "No failures recorded"])
        logging.info(f"Exported failure report to {filename}")
    
    def export_breed_distribution(self, breed_counts, filename="breed_distribution.csv"):
        """Export breed distribution report"""
        headers = ["breed", "image_count"]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for breed, count in breed_counts.items():
                writer.writerow([breed, count])
        logging.info(f"Exported breed distribution report to {filename}")

def generate_erd():
    """Generate ERD file"""
    erd_content = """
    [images]
    id INTEGER PRIMARY KEY AUTOINCREMENT
    url TEXT UNIQUE
    alt_text TEXT
    filename TEXT
    breed TEXT
    """
    erd_file = "images.erd"
    erd_output = "images_erd.png"
    try:
        with open(erd_file, "w", encoding="utf-8") as f:
            f.write(erd_content)
        subprocess.run(["quick-erd", erd_file, "-o", erd_output], check=True)
        logging.info(f"ERD file generated successfully: {erd_output}")
    except subprocess.CalledProcessError as e:
        logging.error(f"ERD generation failed: {str(e)}")
    except FileNotFoundError:
        logging.error("quick-erd not installed, please run 'pip install quick-erd'")
    except Exception as e:
        logging.error(f"Error during ERD generation: {str(e)}")

def main():
    """Main function: execute image collection, cleaning, and reporting"""
    db_manager = None
    cleaner = None
    try:
        # Clear dog_images directory at start
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                file_path = os.path.join(OUTPUT_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logging.info(f"Cleared {OUTPUT_DIR} directory")
        else:
            os.makedirs(OUTPUT_DIR)
            logging.info(f"Created {OUTPUT_DIR} directory")

        # Initialize database
        db_manager = DatabaseManager(DB_NAME)
        
        # Collect training data
        logging.info("Starting training data collection")
        real_dog_collector = ImageCollector(os.path.join(TRAINING_DATA_DIR, "real_dogs"), db_manager)
        real_dog_collector.collect_training_data("real dog photo", max_images=TRAINING_IMAGES_PER_CLASS)
        non_real_dog_collector = ImageCollector(os.path.join(TRAINING_DATA_DIR, "non_real_dogs"), db_manager)
        non_real_keywords = ["cartoon dog", "dog tattoo"]  # Simplified to match old script
        for keyword in non_real_keywords:
            non_real_dog_collector.collect_training_data(keyword, max_images=TRAINING_IMAGES_PER_CLASS // 2, is_non_real=True)
        logging.info(f"Training data collection completed: real_dogs={len(os.listdir(os.path.join(TRAINING_DATA_DIR, 'real_dogs')))}, non_real_dogs={len(os.listdir(os.path.join(TRAINING_DATA_DIR, 'non_real_dogs')))}")
        
        # Initialize and train classifier
        classifier = ImageClassifier(train_data_dir=TRAINING_DATA_DIR)
        
        # Collect main dataset
        collector = ImageCollector(OUTPUT_DIR, db_manager)
        cleaner = DatasetCleaner(OUTPUT_DIR, db_manager, classifier)
        
        logging.info("Starting image collection")
        total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts = collector.collect_and_process()
        
        logging.info(f"Collection phase completed: Total images collected and processed: {total_images}")
        logging.info(f"Images filtered out due to invalid alt text: {filtered_out_alt}")
        logging.info(f"Images failed to download: {download_failed}")
        logging.info(f"Images failed to process: {process_failed}")
        logging.info(f"Images skipped due to duplicate URLs: {duplicate_urls}")
        logging.info(f"Collection phase breed distribution: {breed_counts}")
        print(f"Collection phase completed:")
        print(f"Total images collected and processed: {total_images}")
        print(f"Images filtered out due to invalid alt text: {filtered_out_alt}")
        print(f"Images failed to download: {download_failed}")
        print(f"Images failed to process: {process_failed}")
        print(f"Images skipped due to duplicate URLs: {duplicate_urls}")
        print(f"Collection phase breed distribution: {breed_counts}")
        
        logging.info("Starting dataset cleaning and reporting")
        report, breed_counts = cleaner.generate_report()
        
        cleaner.supplement_dataset(collector)
        
        final_count = db_manager.get_image_count()
        logging.info(f"Final database validation: Stored {final_count} images")
        print(f"Final database validation: Stored {final_count} images")
        
        # Verify file counts
        dog_images_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
        logging.info(f"Final file count in {OUTPUT_DIR}: {dog_images_count}")
        print(f"Final file count in {OUTPUT_DIR}: {dog_images_count}")
        if dog_images_count != final_count:
            logging.warning(f"Mismatch: Database has {final_count} images, but {OUTPUT_DIR} has {dog_images_count} images")
        
        cleaner.failure_log.extend(collector.failure_log)
        cleaner.export_failure_report()
        
        # Generate ERD
        generate_erd()
        
    except Exception as e:
        logging.error(f"Main program error: {str(e)}")
        if cleaner is not None:
            cleaner.failure_log.append(["N/A", "N/A", "N/A", "main_loop", str(e)])
        cleaner.export_failure_report()
    finally:
        if db_manager is not None:
            db_manager.close()

if __name__ == "__main__":
    main()