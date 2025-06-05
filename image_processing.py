import os
import sqlite3
import requests
import logging
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

# 配置日誌記錄
logging.basicConfig(
    filename='image_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置常數
OUTPUT_DIR = "dog_images"
TRAINING_DATA_DIR = "training_data"
DB_NAME = "dog_images.db"
TARGET_IMAGE_COUNT_MIN = 1000
TARGET_IMAGE_COUNT_MAX = 5000
IMAGES_PER_BREED = 400
MAX_IMAGE_SIZE = 100 * 1024
QUALITY_RANGE = (40, 90)
IMAGE_SIZE = (500, 500)
CLASSIFIER_IMAGE_SIZE = (224, 224)
MAX_IMAGES_PER_KEYWORD = 1000
TRAINING_IMAGES_PER_CLASS = 100  # 每個類別（真實狗、非真實狗）收集100張圖片

# 狗品種列表
DOG_BREEDS = [
    "Maltese", "Yorkshire Terrier", "Pomeranian", "Chihuahua", "Miniature Schnauzer",
    "Shih Tzu", "Poodle", "Dachshund", "Shiba Inu", "Labrador Retriever"
]

# 創建輸出目錄和訓練數據目錄
for directory in [OUTPUT_DIR, os.path.join(TRAINING_DATA_DIR, "real_dogs"), os.path.join(TRAINING_DATA_DIR, "non_real_dogs")]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

class DatabaseManager:
    """管理 SQLite 數據庫的操作"""
    def __init__(self, db_name):
        """初始化數據庫連接"""
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.init_database()
    
    def init_database(self):
        """初始化數據庫，創建 images 表"""
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
        """獲取所有圖片記錄"""
        self.cursor.execute("SELECT id, url, alt_text, filename, breed FROM images")
        return self.cursor.fetchall()
    
    def delete_image(self, image_id):
        """從數據庫中刪除圖片記錄"""
        self.cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        self.conn.commit()
    
    def get_image_count(self):
        """獲取數據庫中圖片總數"""
        self.cursor.execute("SELECT COUNT(*) FROM images")
        return self.cursor.fetchone()[0]
    
    def get_unique_domains(self):
        """獲取圖片來源的唯一網域"""
        self.cursor.execute("SELECT url FROM images")
        urls = [row[0] for row in self.cursor.fetchall()]
        domains = {urlparse(url).netloc for url in urls}
        return domains
    
    def get_page_count(self):
        """模擬爬取頁數（基於URL數量估計）"""
        self.cursor.execute("SELECT COUNT(DISTINCT url) FROM images")
        return self.cursor.fetchone()[0] // 50 + 1
    
    def close(self):
        """關閉數據庫連接"""
        self.conn.commit()
        self.conn.close()

class ImageCollector:
    """負責從 Google 和 Bing 收集圖片並下載"""
    def __init__(self, output_dir, db_manager):
        """初始化圖片收集器"""
        self.output_dir = output_dir
        self.db_manager = db_manager
        self.failure_log = []
    
    def get_keyword_variants(self, breed):
        """生成狗品種的關鍵字變體，優先真實狗照片"""
        return [
            f"{breed} real dog photo",
            f"{breed} dog photograph",
            f"{breed} puppy photo",
            f"real {breed} portrait",
            f"{breed} pet photo",
            f"{breed} breed photo"
        ]
    
    def filter_alt_text(self, alt):
        """過濾無效或不相關的 alt 文本，排除卡通和紋身圖片"""
        if not alt or not alt.strip():
            logging.warning(f"Filtered out image with empty alt text")
            return False
        if len(alt.strip()) < 3:
            logging.warning(f"Filtered out image with short alt text: {alt}")
            return False
        blacklist = ["cartoon", "tattoo", "illustration", "drawing", "animated", "artwork"]
        if any(keyword in alt.lower() for keyword in blacklist):
            logging.warning(f"Filtered out image with blacklisted alt text: {alt}")
            return False
        return True
    
    def collect_image_urls_google(self, page, keyword, max_images):
        """從 Google Images 收集圖片 URL 和 alt 文本"""
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
                        if self.filter_alt_text(alt) and (src, alt) not in images:
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
    
    def collect_image_urls_bing(self, page, keyword, max_images):
        """從 Bing Images 收集圖片 URL 和 alt 文本"""
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
                        if self.filter_alt_text(alt) and (src, alt) not in images:
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
    
    def collect_image_urls(self, keyword, max_images):
        """從 Google 和 Bing 收集圖片，去重後返回"""
        images = []
        total_empty_alt_count = 0
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                google_images, google_empty_alt = self.collect_image_urls_google(page, keyword, int(max_images * 2))
                images.extend(google_images)
                total_empty_alt_count += google_empty_alt
                bing_images, bing_empty_alt = self.collect_image_urls_bing(page, keyword, int(max_images * 2))
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
        """下載圖片並保存到本地"""
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
    
    def process_image(self, input_path, output_path):
        """調整圖片大小、裁剪並壓縮為 JPEG"""
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
    
    def collect_training_data(self, keyword, max_images):
        """收集訓練數據並保存到指定目錄"""
        images, empty_alt_count = self.collect_image_urls(keyword, max_images)
        for i, (url, _) in enumerate(images):
            filename = os.path.join(self.output_dir, f"{keyword.replace(' ', '_')}_{i}.jpg")
            if not self.download_image(url, filename):
                self.failure_log.append([keyword, url, "N/A", "download", "Failed after 3 attempts"])
                continue
            if not self.process_image(filename, filename):
                self.failure_log.append([keyword, url, "N/A", "processing", "Compression or processing error"])
                os.remove(filename) if os.path.exists(filename) else None
                continue
            logging.info(f"Collected and processed training image: {filename} for keyword: {keyword}")
    
    def collect_and_process(self):
        """收集並處理圖片，存儲到數據庫"""
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
                    if not self.download_image(url, temp_filename):
                        download_failed += 1
                        self.failure_log.append([keyword, url, alt, "download", "Failed after 3 attempts"])
                        continue
                    if not self.process_image(temp_filename, filename):
                        process_failed += 1
                        self.failure_log.append([keyword, url, alt, "processing", "Compression or processing error"])
                        os.remove(temp_filename) if os.path.exists(temp_filename) else None
                        continue
                    self.db_manager.cursor.execute(
                        "INSERT OR IGNORE INTO images (url, alt_text, filename, breed) VALUES (?, ?, ?, ?)",
                        (url, alt, filename, breed)
                    )
                    if self.db_manager.cursor.rowcount == 0:
                        duplicate_urls += 1
                        self.failure_log.append([keyword, url, alt, "database", "Duplicate URL"])
                    else:
                        total_images += 1
                        breed_counts[breed] += 1
                        logging.info(f"Successfully processed image: {filename} for breed: {breed}")
                    os.remove(temp_filename) if os.path.exists(temp_filename) else None
                self.db_manager.conn.commit()
                logging.info(f"Completed keyword {keyword}, total images: {total_images}, breed {breed}: {breed_counts[breed]}")
        
        return total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts

class ImageClassifier:
    """圖片分類器，基於微調的 MobileNetV2 判斷圖片是否為真實狗照片"""
    def __init__(self, train_data_dir=None):
        """初始化並微調 MobileNetV2 模型"""
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
            logging.info(f"訓練完成：最終訓練準確率 {history.history['accuracy'][-1]:.4f}, "
                        f"驗證準確率 {history.history['val_accuracy'][-1]:.4f}")
    
    def load_training_data(self, data_dir, target_size=(224, 224), batch_size=32):
        """載入訓練數據"""
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
        """繪製訓練損失和準確率曲線"""
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
        logging.info(f"訓練歷史圖表保存至 {output_file}")
    
    def preprocess_image(self, image_path):
        """預處理圖片以適配 MobileNetV2"""
        try:
            img = load_img(image_path, target_size=CLASSIFIER_IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            return img_array
        except Exception as e:
            logging.error(f"圖片預處理失敗 {image_path}: {str(e)}")
            return None
    
    def is_real_dog_image(self, image_path):
        """判斷圖片是否為真實狗照片"""
        try:
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return False
            pred = self.model.predict(img_array)[0][0]
            if 0.5 <= pred < 0.6:
                logging.warning(f"Low confidence real dog image: {image_path}, prob: {pred}")
            return pred > 0.5
        except Exception as e:
            logging.error(f"圖片分類失敗 {image_path}: {str(e)}")
            return False

class DatasetCleaner:
    """負責清理數據集並生成報告"""
    def __init__(self, output_dir, db_manager, classifier):
        """初始化數據集清理器"""
        self.output_dir = output_dir
        self.db_manager = db_manager
        self.classifier = classifier
        self.failure_log = []
    
    def detect_duplicates(self, image_files):
        """檢測重複圖片（基於簡單哈希比較）"""
        hashes = {}
        duplicates = []
        for filename in image_files:
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
                logging.error(f"檢測重複圖片失敗 {filename}: {str(e)}")
                self.failure_log.append(["N/A", filename, "N/A", "duplicate_detection", str(e)])
        return duplicates
    
    def clean_dataset(self):
        """清理數據集，移除不相關和重複圖片"""
        images = self.db_manager.get_all_images()
        initial_count = len(images)
        removed_count = 0
        cartoon_tattoo_count = 0
        image_files = [os.path.join(self.output_dir, row[3]) for row in images]
        
        duplicates = self.detect_duplicates(image_files)
        for dup in duplicates:
            image_id = next(row[0] for row in images if os.path.join(self.output_dir, row[3]) == dup)
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
        
        final_count = self.db_manager.get_image_count()
        logging.info(f"初始圖片數: {initial_count}, 移除圖片數: {removed_count}, 卡通/紋身圖片數: {cartoon_tattoo_count}, 最終圖片數: {final_count}")
        print(f"數據報告：卡通/紋身圖片移除數: {cartoon_tattoo_count}")
        return initial_count, removed_count, final_count, cartoon_tattoo_count
    
    def supplement_dataset(self, collector):
        """補充數據集至 TARGET_IMAGE_COUNT_MIN"""
        final_count = self.db_manager.get_image_count()
        if final_count < TARGET_IMAGE_COUNT_MIN:
            logging.info(f"圖片數 {final_count} 小於最小要求 {TARGET_IMAGE_COUNT_MIN}，進行補充")
            total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts = collector.collect_and_process()
            return total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts
        return None, None, None, None, None, None
    
    def generate_report(self):
        """生成數據報告"""
        initial_count, removed_count, final_count, cartoon_tattoo_count = self.clean_dataset()
        unique_domains = self.db_manager.get_unique_domains()
        page_count = self.db_manager.get_page_count()
        breed_counts = {breed: 0 for breed in DOG_BREEDS}
        for breed in DOG_BREEDS:
            self.db_manager.cursor.execute("SELECT COUNT(*) FROM images WHERE breed = ?", (breed,))
            breed_counts[breed] = self.cursor.fetchone()[0]
        
        report = {
            "初始圖片數": initial_count,
            "移除圖片數": removed_count,
            "卡通/紋身移除數": cartoon_tattoo_count,
            "最終圖片數": final_count,
            "爬取頁數": page_count,
            "唯一網域數": len(unique_domains),
            "品種分佈": breed_counts
        }
        logging.info(f"數據報告: {report}")
        print(f"數據報告: {report}")
        
        self.export_failure_report()
        self.export_breed_distribution(breed_counts)
        
        return report, breed_counts
    
    def export_failure_report(self, filename="image_processing_failure_report.csv"):
        """導出清理失敗報告"""
        headers = ["keyword/breed", "url", "alt_text", "failure_stage", "failure_reason"]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.failure_log)
        logging.info(f"導出失敗報告至 {filename}")
    
    def export_breed_distribution(self, breed_counts, filename="breed_distribution.csv"):
        """導出品種分佈報告"""
        headers = ["breed", "image_count"]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for breed, count in breed_counts.items():
                writer.writerow([breed, count])
        logging.info(f"導出品種分佈報告至 {filename}")

def generate_erd():
    """生成 ERD 文件"""
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
        logging.info(f"ERD 文件生成成功: {erd_output}")
    except subprocess.CalledProcessError as e:
        logging.error(f"ERD 生成失敗: {str(e)}")
    except FileNotFoundError:
        logging.error("quick-erd 未安裝，請運行 'pip install quick-erd'")
    except Exception as e:
        logging.error(f"ERD 生成過程中發生錯誤: {str(e)}")

def main():
    """主函數：執行圖片收集、清理和報告生成"""
    db_manager = None
    cleaner = None
    try:
        # 初始化數據庫
        db_manager = DatabaseManager(DB_NAME)
        
        # 收集訓練數據
        logging.info("開始收集訓練數據")
        real_dog_collector = ImageCollector(os.path.join(TRAINING_DATA_DIR, "real_dogs"), db_manager)
        real_dog_collector.collect_training_data("real dog photo", max_images=TRAINING_IMAGES_PER_CLASS)
        non_real_dog_collector = ImageCollector(os.path.join(TRAINING_DATA_DIR, "non_real_dogs"), db_manager)
        non_real_dog_collector.collect_training_data("cartoon dog", max_images=TRAINING_IMAGES_PER_CLASS // 2)
        non_real_dog_collector.collect_training_data("dog tattoo", max_images=TRAINING_IMAGES_PER_CLASS // 2)
        logging.info("訓練數據收集完成")
        
        # 初始化分類器並訓練
        classifier = ImageClassifier(train_data_dir=TRAINING_DATA_DIR)
        
        # 收集主數據集
        collector = ImageCollector(OUTPUT_DIR, db_manager)
        cleaner = DatasetCleaner(OUTPUT_DIR, db_manager, classifier)
        
        logging.info("開始圖片收集")
        total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts = collector.collect_and_process()
        
        logging.info(f"收集階段完成：總共收集並處理的圖片數：{total_images}")
        logging.info(f"因無效alt文本過濾掉的圖片：{filtered_out_alt}")
        logging.info(f"下載失敗的圖片：{download_failed}")
        logging.info(f"處理失敗的圖片：{process_failed}")
        logging.info(f"重複URL跳過的圖片：{duplicate_urls}")
        logging.info(f"收集階段品種分佈：{breed_counts}")
        print(f"收集階段完成：")
        print(f"總共收集並處理的圖片數：{total_images}")
        print(f"因無效alt文本過濾掉的圖片：{filtered_out_alt}")
        print(f"下載失敗的圖片：{download_failed}")
        print(f"處理失敗的圖片：{process_failed}")
        print(f"重複URL跳過的圖片：{duplicate_urls}")
        print(f"收集階段品種分佈：{breed_counts}")
        
        logging.info("開始數據集清理和報告生成")
        report, breed_counts = cleaner.generate_report()
        
        cleaner.supplement_dataset(collector)
        
        final_count = db_manager.get_image_count()
        logging.info(f"最終數據庫驗證：存儲了 {final_count} 張圖片")
        print(f"最終數據庫驗證：存儲了 {final_count} 張圖片")
        
        cleaner.failure_log.extend(collector.failure_log)
        cleaner.export_failure_report()
        
        # 生成 ERD
        generate_erd()
        
    except Exception as e:
        logging.error(f"主程序錯誤: {str(e)}")
        if cleaner is not None:
            cleaner.failure_log.append(["N/A", "N/A", "N/A", "main_loop", str(e)])
    finally:
        if db_manager is not None:
            db_manager.close()

if __name__ == "__main__":
    main()