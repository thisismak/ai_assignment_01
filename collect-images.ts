import { chromium, Page } from '@playwright/test';
import axios from 'axios';
import sharp from 'sharp';
import sqlite3 from 'sqlite3';
import { existsSync, mkdirSync, unlinkSync, writeFileSync, appendFileSync } from 'fs';
import { join } from 'path';

// 配置
const OUTPUT_DIR = 'dog_images';
const DB_PATH = 'db.sqlite3'; // 與 server.ts 共用
const LOG_PATH = 'log.txt';
const MAX_IMAGES = 5000;
const MIN_IMAGES = 3000;
const MAX_SIZE_KB = 50;
const TARGET_SIZE = { width: 500, height: 500 };
const QUALITY_RANGE = { min: 50, max: 80 };
const KEYWORD = 'various dog breeds';

// 日誌函數（使用香港時間）
function logMessage(message: string): void {
  const timestamp = new Date().toLocaleString('zh-HK', {
    timeZone: 'Asia/Hong_Kong',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3,
    hour12: false,
  });
  const logEntry = `${timestamp} - ${message}\n`;
  console.log(message);
  appendFileSync(LOG_PATH, logEntry);
}

// 確保輸出資料夾存在
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// 圖像數據接口
interface ImageData {
  src: string;
  alt: string | null;
  filename: string | null;
}

// 初始化資料庫
function initDatabase(callback: (err: Error | null, db: sqlite3.Database) => void): void {
  const db = new sqlite3.Database(DB_PATH, (err) => {
    if (err) {
      logMessage(`Database connection error: ${err.message}`);
      callback(err, db);
      return;
    }
    logMessage('Connected to SQLite database at ' + DB_PATH);
    db.run(`
      CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        src TEXT NOT NULL,
        alt TEXT,
        filename TEXT
      )
    `, (err) => {
      if (err) {
        logMessage(`Table creation error: ${err.message}`);
        callback(err, db);
        return;
      }
      logMessage('Initialized images table');
      callback(null, db);
    });
  });
}

// 搜集圖像元數據
async function searchImages(page: Page, keyword: string, maxImages: number): Promise<ImageData[]> {
  logMessage(`Starting image search for keyword: "${keyword}"`);
  const imagesData: ImageData[] = [];

  await page.setExtraHTTPHeaders({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
  });
  await page.setViewportSize({ width: 1280, height: 720 });

  try {
    await page.goto(`https://www.google.com/search?q=${encodeURIComponent(keyword)}&tbm=isch`, {
      waitUntil: 'domcontentloaded',
      timeout: 30000,
    });
    logMessage('Successfully loaded Google Images page');
    await page.screenshot({ path: 'google_images_screenshot.png' });
  } catch (error: any) {
    logMessage(`Failed to load Google Images page: ${error.message}`);
    return imagesData;
  }

  let lastCount = 0;
  let scrollAttempts = 0;
  const maxScrollAttempts = 20;

  while (imagesData.length < maxImages && scrollAttempts < maxScrollAttempts) {
    const moreButton = await page.$('input[value="Show more"]');
    if (moreButton) {
      logMessage('Clicking "Show more" button');
      await moreButton.click();
      await page.waitForTimeout(5000);
    } else {
      await page.evaluate('window.scrollBy(0, 1000)');
      await page.waitForTimeout(5000);
    }

    const imgElements = await page.$$('img');
    logMessage(`Found ${imgElements.length} img elements on page`);
    for (const img of imgElements) {
      const src = (await img.getAttribute('data-src')) || (await img.getAttribute('src'));
      const alt = await img.getAttribute('alt');
      logMessage(`Checking image: src=${src}, alt=${alt}`);
      if (src && (src.startsWith('http') || src.startsWith('https')) && imagesData.length < maxImages) {
        imagesData.push({ src, alt: alt || null, filename: null });
      }
    }

    if (imagesData.length === lastCount && !moreButton) {
      logMessage('No more images to load');
      break;
    }
    lastCount = imagesData.length;
    scrollAttempts++;
  }

  logMessage(`Collected ${imagesData.length} image metadata entries`);
  return imagesData;
}

// 下載圖像
async function downloadImage(src: string, filename: string): Promise<boolean> {
  try {
    const response = await axios.get(src, { responseType: 'arraybuffer', timeout: 10000 });
    if (response.status === 200) {
      writeFileSync(filename, response.data);
      logMessage(`Downloaded image: ${filename}`);
      return true;
    }
    logMessage(`Failed to download image: ${src} (status: ${response.status})`);
    return false;
  } catch (err: any) {
    logMessage(`Download image ${src} failed: ${err.message}`);
    return false;
  }
}

// 處理圖像
async function processImage(inputPath: string, outputPath: string): Promise<boolean> {
  try {
    let quality = QUALITY_RANGE.max;
    while (quality >= QUALITY_RANGE.min) {
      const buffer = await sharp(inputPath)
        .resize(TARGET_SIZE.width, TARGET_SIZE.height, { fit: 'cover', position: 'center' })
        .jpeg({ quality })
        .toBuffer();
      const sizeKB = buffer.length / 1024;
      if (sizeKB <= MAX_SIZE_KB) {
        await sharp(buffer).toFile(outputPath);
        logMessage(`Processed image: ${outputPath} (${sizeKB.toFixed(2)}KB)`);
        return true;
      }
      quality -= 5;
    }
    logMessage(`Image ${inputPath} could not be compressed below ${MAX_SIZE_KB}KB`);
    return false;
  } catch (err: any) {
    logMessage(`Process image ${inputPath} failed: ${err.message}`);
    return false;
  }
}

// 主函數
async function main() {
  logMessage('Starting image collection and processing');
  initDatabase(async (err, db) => {
    if (err) {
      logMessage(`Failed to initialize database: ${err.message}`);
      return;
    }

    try {
      const browser = await chromium.launch({ headless: false });
      const page = await browser.newPage();
      const imagesData = await searchImages(page, KEYWORD, MAX_IMAGES);
      await browser.close();

      if (imagesData.length < MIN_IMAGES) {
        logMessage(`Collected ${imagesData.length} images, less than required ${MIN_IMAGES}`);
        db.close((err) => {
          if (err) logMessage(`Database close error: ${err.message}`);
        });
        return;
      }

      let processedCount = 0;
      db.run('BEGIN TRANSACTION', (err) => {
        if (err) {
          logMessage(`Transaction start error: ${err.message}`);
          db.close((err) => {
            if (err) logMessage(`Database close error: ${err.message}`);
          });
          return;
        }

        const processNextImage = async (index: number) => {
          if (index >= imagesData.length || processedCount >= MAX_IMAGES) {
            db.run('COMMIT', (err) => {
              if (err) {
                logMessage(`Transaction commit error: ${err.message}`);
                db.close((err) => {
                  if (err) logMessage(`Database close error: ${err.message}`);
                });
                return;
              }
              db.get('SELECT COUNT(*) as count FROM images', (err, result: { count: number }) => {
                if (err) {
                  logMessage(`Count query error: ${err.message}`);
                  db.close((err) => {
                    if (err) logMessage(`Database close error: ${err.message}`);
                  });
                  return;
                }
                logMessage(`Successfully processed ${processedCount} images. Total in database: ${result.count}`);
                db.close((err) => {
                  if (err) logMessage(`Database close error: ${err.message}`);
                });
              });
            });
            return;
          }

          const image = imagesData[index];
          const filename = join(OUTPUT_DIR, `dog_${processedCount + 1}.jpg`);
          const tempFilename = join(OUTPUT_DIR, `temp_${processedCount + 1}.jpg`);

          if (await downloadImage(image.src, tempFilename)) {
            if (await processImage(tempFilename, filename)) {
              db.run('INSERT INTO images (src, alt, filename) VALUES (?, ?, ?)', [image.src, image.alt, filename], (err) => {
                if (err) {
                  logMessage(`Insert image error: ${err.message}`);
                } else {
                  image.filename = filename;
                  processedCount++;
                }
                if (existsSync(tempFilename)) {
                  unlinkSync(tempFilename);
                }
                processNextImage(index + 1);
              });
            } else {
              if (existsSync(tempFilename)) {
                unlinkSync(tempFilename);
              }
              processNextImage(index + 1);
            }
          } else {
            processNextImage(index + 1);
          }
        };

        processNextImage(0);
      });
    } catch (err: any) {
      logMessage(`Main execution failed: ${err.message}`);
      db.run('ROLLBACK', () => {
        db.close((err) => {
          if (err) logMessage(`Database close error: ${err.message}`);
        });
      });
    }
  });
}

// 執行主函數
main().catch((err: any) => {
  logMessage(`Main execution failed: ${err.message}`);
});