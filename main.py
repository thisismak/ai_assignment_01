import subprocess
import os
import logging
import sys
import importlib.util
import sqlite3
from pathlib import Path

# Configure logging for tracking progress and errors
logging.basicConfig(
    filename='main.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Configuration constants
SCRIPT_PY = "script.py"
CLEANER_PY = "image_dataset_cleaner.py"
DB_NAME = "dog_images.db"
OUTPUT_DIR = "dog_images"
CLEANED_DIR = "cleaned_dog_images"
PAGE_COUNT_LOG = "image_collection.log"
EXPECTED_SCRIPT_IMAGES = (3000, 5000)  # Expected range for script.py
EXPECTED_CLEANED_IMAGES = (1000, 2000)  # Expected range for image_dataset_cleaner.py

def check_dependencies():
    """Check if required Python modules are installed."""
    required_modules = {
        'tensorflow': 'tensorflow',
        'pillow': 'PIL',
        'imagehash': 'imagehash',
        'numpy': 'numpy',
        'requests': 'requests',
        'playwright': 'playwright'
    }
    missing_modules = []
    for package_name, import_name in required_modules.items():
        try:
            importlib.util.find_spec(import_name)
            logging.info(f"Module {package_name} (imported as {import_name}) found")
        except ImportError:
            missing_modules.append(package_name)
            logging.error(f"Module {package_name} (imported as {import_name}) not found")
    if missing_modules:
        logging.error(f"Missing required modules: {', '.join(missing_modules)}")
        print(f"錯誤：缺少以下模組：{', '.join(missing_modules)}")
        print("請使用以下命令安裝：")
        print(f"python -m pip install {' '.join(missing_modules)}")
        if 'playwright' in missing_modules:
            print("playwright 還需要額外安裝瀏覽器依賴：")
            print("playwright install")
        return False
    logging.info("All required modules are installed")
    return True

def count_images(directory):
    """Count the number of image files in a directory."""
    try:
        if not os.path.exists(directory):
            logging.warning(f"Directory {directory} does not exist")
            return 0
        return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    except Exception as e:
        logging.error(f"Error counting images in {directory}: {str(e)}")
        return 0

def count_db_records(db_name):
    """Count the number of records in the images table."""
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
        if not cursor.fetchone():
            logging.error(f"No 'images' table found in {db_name}")
            conn.close()
            return 0
        cursor.execute("SELECT COUNT(*) FROM images")
        count = cursor.fetchone()[0]
        conn.close()
        logging.info(f"Database {db_name} contains {count} records")
        return count
    except sqlite3.OperationalError as e:
        logging.error(f"Database error counting records: {str(e)}")
        return 0
    except Exception as e:
        logging.error(f"Unexpected error counting DB records: {str(e)}")
        return 0

def validate_db_files(db_name, image_dir):
    """Validate that database records have corresponding image files."""
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM images")
        files = cursor.fetchall()
        conn.close()
        missing_files = [f[0] for f in files if not os.path.exists(f[0])]
        if missing_files:
            logging.warning(f"Found {len(missing_files)} missing image files in {image_dir}")
            for f in missing_files[:5]:
                logging.warning(f"Missing file: {f}")
            print(f"警告：找到 {len(missing_files)} 個缺失的圖像文件，示例：{missing_files[:5]}")
        return len(missing_files) == 0
    except Exception as e:
        logging.error(f"Error validating database files: {str(e)}")
        return False

def log_and_print_stats(script_name, image_dir, expected_range):
    """Log and print image and DB record counts."""
    image_count = count_images(image_dir)
    db_count = count_db_records(DB_NAME)  # Fixed: Changed db_name to DB_NAME
    logging.info(f"{script_name} 執行結果：")
    logging.info(f"- 圖像數量（{image_dir}）：{image_count}")
    logging.info(f"- SQL 記錄數量（{DB_NAME}）：{db_count}")
    print(f"{script_name} 執行結果：")
    print(f"- 圖像數量（{image_dir}）：{image_count}")
    print(f"- SQL 記錄數量（{DB_NAME}）：{db_count}")
    
    if not expected_range[0] <= image_count <= expected_range[1]:
        logging.warning(f"{script_name} 圖像數量 {image_count} 不在預期範圍 {expected_range}")
        print(f"警告：{script_name} 圖像數量 {image_count} 不在預期範圍 {expected_range}")
    if image_count != db_count:
        logging.warning(f"{script_name} 圖像數量 {image_count} 與 SQL 記錄數量 {db_count} 不一致")
        print(f"警告：{script_name} 圖像數量 {image_count} 與 SQL 記錄數量 {db_count} 不一致")
    
    return image_count, db_count

def run_script(script_name):
    """Run a Python script using subprocess and handle errors."""
    try:
        result = subprocess.run([sys.executable, script_name], capture_output=True, text=True, check=True)
        logging.info(f"Successfully ran {script_name}")
        print(f"成功運行 {script_name}")
        print(result.stdout)
        if result.stderr:
            logging.warning(f"Warnings/Errors from {script_name}: {result.stderr}")
            print(f"來自 {script_name} 的警告/錯誤：{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run {script_name}: {e}")
        logging.error(f"STDERR: {e.stderr}")
        print(f"錯誤：無法運行 {script_name}：{e}")
        print(f"詳細錯誤：{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error running {script_name}: {str(e)}")
        print(f"意外錯誤：運行 {script_name} 時發生 {str(e)}")
        return False

def check_prerequisites():
    """Check if prerequisites for image_dataset_cleaner.py are met."""
    if not os.path.exists(SCRIPT_PY):
        logging.error(f"Script file {SCRIPT_PY} not found in current directory")
        print(f"錯誤：{SCRIPT_PY} 文件不存在於當前目錄")
        return False
    if not os.path.exists(CLEANER_PY):
        logging.error(f"Script file {CLEANER_PY} not found in current directory")
        print(f"錯誤：{CLEANER_PY} 文件不存在於當前目錄")
        return False
    if not os.path.exists(DB_NAME):
        logging.error(f"Database file {DB_NAME} not found. Please ensure {SCRIPT_PY} runs successfully.")
        print(f"錯誤：數據庫文件 {DB_NAME} 不存在。請確保 {SCRIPT_PY} 成功運行。")
        return False
    if not os.path.exists(OUTPUT_DIR):
        logging.error(f"Image directory {OUTPUT_DIR} not found. Please ensure {SCRIPT_PY} runs successfully.")
        print(f"錯誤：圖像目錄 {OUTPUT_DIR} 不存在。請確保 {SCRIPT_PY} 成功運行。")
        return False
    if not os.path.exists(PAGE_COUNT_LOG):
        logging.warning(f"Log file {PAGE_COUNT_LOG} not found. Page count will be set to 0.")
        print(f"警告：日誌文件 {PAGE_COUNT_LOG} 不存在。爬取頁數將設為 0。")
    
    # Check if database has records
    db_count = count_db_records(DB_NAME)
    if db_count == 0:
        logging.error("No records in images table. Please ensure script.py generates data.")
        print("錯誤：數據庫 images 表無記錄。請確保 script.py 成功生成數據。")
        return False
    
    # Validate database files
    if not validate_db_files(DB_NAME, OUTPUT_DIR):
        logging.error("Database contains references to missing image files. Cleaning may fail.")
        print("錯誤：數據庫包含缺失圖像文件的引用，清理可能失敗。")
        return False
    
    return True

def main():
    """Main function to orchestrate execution of script.py and image_dataset_cleaner.py."""
    logging.info("Starting one-click execution of AI Assignment")
    print("開始一鍵執行 AI 作業...")
    print(f"使用 Python 版本：{sys.version}")
    print(f"Python 可執行路徑：{sys.executable}")

    # Check dependencies
    if not check_dependencies():
        logging.error("Aborting due to missing dependencies")
        print("因缺少依賴而中止")
        return

    # Check if script files exist
    if not os.path.exists(SCRIPT_PY) or not os.path.exists(CLEANER_PY):
        logging.error("One or both script files are missing")
        print("錯誤：缺少一個或兩個腳本文件")
        return

    # Run Exercise 1: script.py
    logging.info(f"Running {SCRIPT_PY} to collect images")
    print(f"正在運行 {SCRIPT_PY} 以收集圖像...")
    if not run_script(SCRIPT_PY):
        logging.error("Aborting due to failure in script.py")
        print(f"錯誤：{SCRIPT_PY} 運行失敗，中止後續操作")
        return

    # Log and check script.py results
    script_images, script_db = log_and_print_stats(SCRIPT_PY, OUTPUT_DIR, EXPECTED_SCRIPT_IMAGES)
    if script_images == 0 or script_db == 0:
        logging.error("No images or DB records generated by script.py. Aborting.")
        print("錯誤：script.py 未生成圖像或數據庫記錄，中止後續操作")
        return

    # Check prerequisites for Exercise 2
    if not check_prerequisites():
        logging.error("Aborting due to missing prerequisites")
        print("錯誤：缺少前置條件，中止後續操作")
        return

    # Run Exercise 2: image_dataset_cleaner.py
    logging.info(f"Running {CLEANER_PY} to clean and analyze dataset")
    print(f"正在運行 {CLEANER_PY} 以清理和分析數據集...")
    if not run_script(CLEANER_PY):
        logging.error("Failed to complete image_dataset_cleaner.py")
        print(f"錯誤：{CLEANER_PY} 運行失敗")
        return

    # Log and check image_dataset_cleaner.py results
    cleaned_images, cleaned_db = log_and_print_stats(CLEANER_PY, CLEANED_DIR, EXPECTED_CLEANED_IMAGES)

    logging.info("One-click execution completed successfully")
    print("一鍵執行成功完成！")
    print("最終輸出：")
    print(f"- script.py：{script_images} 張圖像，{script_db} 條 SQL 記錄")
    print(f"- image_dataset_cleaner.py：{cleaned_images} 張清理後圖像，{cleaned_db} 條 SQL 記錄")
    print("輸出文件：")
    print(f"- 數據庫：{DB_NAME}")
    print(f"- 原始圖像目錄：{OUTPUT_DIR}")
    print(f"- 清理後圖像目錄：{CLEANED_DIR}")
    print(f"- 報告文件：dataset_report.csv, cleaned_breed_distribution.csv, cleaning_failure_report.csv")
    print(f"- 日誌文件：main.log, image_collection.log, dataset_cleaning.log")

if __name__ == "__main__":
    main()