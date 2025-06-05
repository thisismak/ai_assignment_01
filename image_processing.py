import os
import sqlite3
import requests
import logging
import csv
import random
from urllib.parse import urlparse
from uuid import uuid4
from playwright.sync_api import sync_playwright
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# 配置日誌記錄，統一記錄收集和清理過程
logging.basicConfig(
    filename='image_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置常數
OUTPUT_DIR = "dog_images"
DB_NAME = "dog_images.db"
TARGET_IMAGE_COUNT_MIN = 1000  # 習作二的最小圖片數要求
TARGET_IMAGE_COUNT_MAX = 5000  # 習作一和二的共用最大值
IMAGES_PER_BREED = 400  # 每品種最大圖片數
MAX_IMAGE_SIZE = 100 * 1024  # 100KB
QUALITY_RANGE = (40, 90)  # 圖片壓縮質量範圍
IMAGE_SIZE = (500, 500)  # 習作一的圖片尺寸
CLASSIFIER_IMAGE_SIZE = (224, 224)  # MobileNetV2 輸入尺寸
MAX_IMAGES_PER_KEYWORD = 1000  # 每關鍵字最大圖片數

# 狗品種列表
DOG_BREEDS = [
    "Maltese", "Yorkshire Terrier", "Pomeranian", "Chihuahua", "Miniature Schnauzer",
    "Shih Tzu", "Poodle", "Dachshund", "Shiba Inu", "Labrador Retriever"
]

# 創建輸出目錄
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class DatabaseManager:
    """管理 SQLite 數據庫的操作"""
    def __init__(self, db_name):
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
        return self.cursor.fetchone()[0] // 50 + 1  # 假設每頁50張圖片
    
    def close(self):
        """關閉數據庫連接"""
        self.conn.commit()
        self.conn.close()

class ImageCollector:
    """負責從 Google 和 Bing 收集圖片並下載"""
    def __init__(self, output_dir, db_manager):
        self.output_dir = output_dir
        self.db_manager = db_manager
        self.failure_log = []
    
    def get_keyword_variants(self, breed):
        """生成狗品種的關鍵字變體"""
        return [
            breed,
            f"{breed} dog",
            f"{breed} puppy",
            f"cute {breed}",
            f"{breed} portrait",
            f"{breed} pet",
            f"{breed} breed"
        ]
    
    def filter_alt_text(self, alt):
        """過濾無效的 alt 文本"""
        if not alt or not alt.strip():
            logging.warning(f"Filtered out image with empty alt text")
            return False
        if len(alt.strip()) < 3:
            logging.warning(f"Filtered out image with short alt text: {alt}")
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
                images = list(dict.fromkeys(images))  # 去重
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
        
        # 第一輪：每個品種最多 IMAGES_PER_BREED 張圖片
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
        
        # 第二輪：補充圖片至 TARGET_IMAGE_COUNT_MIN
        if total_images < TARGET_IMAGE_COUNT_MIN:
            logging.info(f"Shortfall of {TARGET_IMAGE_COUNT_MIN - total_images} images, retrying underfilled breeds")
            underfilled_breeds = [breed for breed, count in breed_counts.items() if count < IMAGES_PER_BREED]
            random.shuffle(underfilled_breeds)
            for breed in underfilled_breeds:
                if total_images >= TARGET_IMAGE_COUNT_MIN:
                    break
                for keyword in self.get_keyword_variants(breed):
                    if total_images >= TARGET_IMAGE_COUNT_MIN or breed_counts[breed] >= IMAGES_PER_BREED:
                        break
                    logging.info(f"Supplementing keyword {keyword} for breed {breed}")
                    images, empty_alt_count = self.collect_image_urls(keyword, MAX_IMAGES_PER_KEYWORD)
                    filtered_out_alt += empty_alt_count
                    for i, (url, alt) in enumerate(images):
                        if total_images >= TARGET_IMAGE_COUNT_MIN or breed_counts[breed] >= IMAGES_PER_BREED:
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
                    logging.info(f"Completed supplementing keyword {keyword}, total images: {total_images}, breed {breed}: {breed_counts[breed]}")
        
        return total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts

class ImageClassifier:
    """圖片分類器，基於 MobileNetV2 判斷圖片是否為狗"""
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet')
    
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
    
    def is_dog_image(self, image_path):
        """判斷圖片是否包含狗"""
        try:
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return False
            preds = self.model.predict(img_array)
            decoded = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=3)[0]
            for _, label, prob in decoded:
                if 'dog' in label.lower() and prob > 0.5:  # 置信度閾值
                    return True
            return False
        except Exception as e:
            logging.error(f"圖片分類失敗 {image_path}: {str(e)}")
            return False

class DatasetCleaner:
    """負責清理數據集並生成報告"""
    def __init__(self, output_dir, db_manager):
        self.output_dir = output_dir
        self.db_manager = db_manager
        self.classifier = ImageClassifier()
        self.failure_log = []
    
    def detect_duplicates(self, image_files):
        """檢測重複圖片（基於圖片內容的簡單哈希比較）"""
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
        """清理數據集：移除不相關和重複圖片"""
        images = self.db_manager.get_all_images()
        initial_count = len(images)
        removed_count = 0
        image_files = [os.path.join(self.output_dir, row[3]) for row in images]
        
        # 檢測重複圖片
        duplicates = self.detect_duplicates(image_files)
        for dup in duplicates:
            image_id = next(row[0] for row in images if os.path.join(self.output_dir, row[3]) == dup)
            self.db_manager.delete_image(image_id)
            if os.path.exists(dup):
                os.remove(dup)
            removed_count += 1
            logging.info(f"移除重複圖片: {dup}")
            self.failure_log.append(["N/A", dup, "N/A", "duplicate_removal", "Duplicate image"])
        
        # 檢查圖片是否為狗的圖像
        for image_id, url, alt_text, filename, breed in images:
            file_path = os.path.join(self.output_dir, filename)
            if not os.path.exists(file_path):
                self.db_manager.delete_image(image_id)
                removed_count += 1
                logging.info(f"移除不存在的圖片: {file_path}")
                self.failure_log.append([breed, url, alt_text, "file_check", "File not found"])
                continue
            if not self.classifier.is_dog_image(file_path):
                self.db_manager.delete_image(image_id)
                if os.path.exists(file_path):
                    os.remove(file_path)
                removed_count += 1
                logging.info(f"移除不相關圖片: {file_path}")
                self.failure_log.append([breed, url, alt_text, "classification", "Not a dog image"])
        
        final_count = self.db_manager.get_image_count()
        logging.info(f"初始圖片數: {initial_count}, 移除圖片數: {removed_count}, 最終圖片數: {final_count}")
        return initial_count, removed_count, final_count
    
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
        initial_count, removed_count, final_count = self.clean_dataset()
        unique_domains = self.db_manager.get_unique_domains()
        page_count = self.db_manager.get_page_count()
        breed_counts = {}
        for breed in DOG_BREEDS:
            self.db_manager.cursor.execute("SELECT COUNT(*) FROM images WHERE breed = ?", (breed,))
            breed_counts[breed] = self.db_manager.cursor.fetchone()[0]
        
        # 導出報告
        self.export_failure_report()
        self.export_breed_distribution(breed_counts)
        
        # 打印報告
        report = {
            "初始圖片數": initial_count,
            "移除圖片數": removed_count,
            "最終圖片數": final_count,
            "爬取頁數": page_count,
            "唯一網域數": len(unique_domains),
            "品種分佈": breed_counts
        }
        logging.info(f"數據報告: {report}")
        print(f"數據報告:")
        print(f"初始圖片數: {initial_count}")
        print(f"移除圖片數: {removed_count}")
        print(f"最終圖片數: {final_count}")
        print(f"爬取頁數: {page_count}")
        print(f"唯一網域數: {len(unique_domains)}")
        print(f"品種分佈: {breed_counts}")
        
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

def main():
    """主函數：執行圖片收集、清理和報告生成"""
    try:
        db_manager = DatabaseManager(DB_NAME)
        collector = ImageCollector(OUTPUT_DIR, db_manager)
        cleaner = DatasetCleaner(OUTPUT_DIR, db_manager)
        
        # 收集圖片（習作一）
        logging.info("開始圖片收集（習作一）")
        total_images, filtered_out_alt, download_failed, process_failed, duplicate_urls, breed_counts = collector.collect_and_process()
        
        # 記錄收集階段的結果
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
        
        # 清理數據集並生成報告（習作二）
        logging.info("開始數據集清理和報告生成（習作二）")
        report, breed_counts = cleaner.generate_report()
        
        # 補充圖片（如果最終圖片數小於 TARGET_IMAGE_COUNT_MIN）
        cleaner.supplement_dataset(collector)
        
        # 驗證最終數據庫計數
        final_count = db_manager.get_image_count()
        logging.info(f"最終數據庫驗證：存儲了 {final_count} 張圖片")
        print(f"最終數據庫驗證：存儲了 {final_count} 張圖片")
        
        # 合併失敗報告
        cleaner.failure_log.extend(collector.failure_log)
        cleaner.export_failure_report()
        
    except Exception as e:
        logging.error(f"主程序錯誤: {str(e)}")
        cleaner.failure_log.append(["N/A", "N/A", "N/A", "main_loop", str(e)])
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()