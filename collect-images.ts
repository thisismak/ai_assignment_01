import { chromium, Page } from '@playwright/test';
import axios from 'axios';
import sharp from 'sharp';
import sqlite3 from 'sqlite3';
import { existsSync, mkdirSync, unlinkSync, writeFileSync, appendFileSync } from 'fs';
import { join } from 'path';

// Configuration
// 配置常量，定義圖像收集的參數和篩選條件
const OUTPUT_DIR = 'dog_images'; // 圖像保存目錄，存儲下載和處理後的圖像
const DB_PATH = 'db.sqlite3'; // SQLite 數據庫文件路徑，存儲搜索記錄和圖像元數據
const LOG_PATH = 'log.txt'; // 日誌文件路徑，記錄搜索、下載、處理的詳細信息
const MAX_IMAGES = 5000; // 最大圖像數量，限制總收集數量
const MIN_IMAGES = 3000; // 最小圖像數量，若少於此數量會記錄警告
const MAX_SIZE_KB = 50; // 處理後圖像的最大文件大小（KB）
const TARGET_SIZE = { width: 500, height: 500 }; // 處理後圖像的目標尺寸（像素）
const QUALITY_RANGE = { min: 50, max: 80 }; // JPEG 壓縮質量範圍
// 狗品種關鍵字清單，用於篩選圖像的 alt 文本
const DOG_BREED_TERMS = [
  'chihuahua', '吉娃娃',
  'pomeranian', '博美犬',
  'yorkshire terrier', '約克夏梗',
  'maltese', '馬爾濟斯犬',
  'dachshund', '臘腸犬',
  'miniature poodle', '迷你貴賓犬', 'poodle', '貴賓犬',
  'french bulldog', '法國鬥牛犬',
  'bichon frise', '比熊犬',
  'shih tzu', '西施犬',
  'miniature schnauzer', '迷你雪納瑞',
  'labrador', '拉布拉多',
  'golden retriever', '黃金獵犬',
  'german shepherd', '德國牧羊犬',
  'bulldog', '英國鬥牛犬',
  'beagle', '比格犬',
  'husky', '哈士奇',
  'rottweiler', '羅威納犬',
  'boxer', '拳師犬',
  'siberian husky', '西伯利亞哈士奇',
  'doberman', '杜賓犬',
  'great dane', '大丹犬',
  'pug', '巴哥犬',
  'border collie', '邊境牧羊犬',
  'australian shepherd', '澳洲牧羊犬',
  '純種狗', '狗狗'
];
const MAX_IMAGES_PER_BREED = 1000; // 每個品種的最大圖像數量，防止單一品種過多
const USER_AGENTS = [
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15',
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0',
];

// Log function (Hong Kong time)
// 日誌記錄函數，將操作詳情寫入 log.txt，包含香港時間戳
// 用途：追蹤搜索、下載、處理的每一步，便於調試圖像數量不足的問題
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

// Ensure output directory exists
// 確保圖像保存目錄存在，若不存在則創建
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  logMessage(`Created output directory: ${OUTPUT_DIR}`);
}

// Image data interface
// 定義圖像數據結構，存儲 src、alt、filename 和品種信息
interface ImageData {
  src: string; // 圖像 URL
  alt: string | null; // 圖像 alt 文本，用於篩選
  filename: string | null; // 保存的文件名
  breed?: string; // 匹配的狗品種，記錄篩選結果
}

// Initialize database
// 初始化 SQLite 數據庫，連接到 db.sqlite3
// 用途：存儲 searches 和 images 表的數據
// 與 server.ts 的連動：server.ts 的 /search 端點通過 main 函數調用此腳本，數據庫操作與 server.ts 共享 db.sqlite3
function initDatabase(callback: (err: Error | null, db: sqlite3.Database) => void): void {
  const db = new sqlite3.Database(DB_PATH, (err) => {
    if (err) {
      logMessage(`Database connection error: ${err.message}`);
      callback(err, db);
      return;
    }
    logMessage('Connected to SQLite database at ' + DB_PATH);
    callback(null, db);
  });
}

// Search images
// 搜索圖像的核心函數，從 Google 圖像搜索收集圖像元數據
// 參數：
// - page: Playwright 瀏覽器頁面，用於爬取 Google 圖像
// - keyword: 搜索關鍵字，來自 main 函數的 keywords 數組
// - maxImages: 本次搜索的最大圖像數量
// - breedCounts: 記錄各品種的圖像數量，限制每品種數量
// 返回：ImageData 數組，包含符合條件的圖像元數據
// 篩選邏輯：僅接受有 alt 文本且匹配 DOG_BREED_TERMS 或包含“dog”/“puppy”的圖像
// 注意：篩選可能過嚴（僅接受有 alt 文本的圖像），導致圖像數量不足
// 與 search.html 的連動：search.html 的表單提交關鍵字，通過 server.ts 的 /search 端點傳遞到 main 函數
async function searchImages(page: Page, keyword: string, maxImages: number, breedCounts: { [key: string]: number }): Promise<ImageData[]> {
  logMessage(`Starting image search for keyword: "${keyword}"`);
  const imagesData: ImageData[] = [];

  // Rotate User-Agent
  // 隨機選擇 User-Agent，模擬不同瀏覽器，減少反爬蟲限制
  const userAgent = USER_AGENTS[Math.floor(Math.random() * USER_AGENTS.length)];
  await page.setExtraHTTPHeaders({ 'user-agent': userAgent });
  await page.setViewportSize({ width: 1280, height: 720 });

  // 嘗試加載 Google 圖像搜索頁面，最多重試 3 次
  let retries = 3;
  while (retries > 0) {
    try {
      await page.goto(`https://www.google.com/search?q=${encodeURIComponent(keyword)}&tbm=isch`, {
        waitUntil: 'domcontentloaded',
        timeout: 30000,
      });
      logMessage('Successfully loaded Google Images page');
      await page.screenshot({ path: `google_images_screenshot_${keyword.replace(/\s+/g, '_')}.jpg` });
      break;
    } catch (error: any) {
      logMessage(`Failed to load Google Images page (Retry ${4 - retries}): ${error.message}`);
      retries--;
      if (retries > 0) await page.waitForTimeout(5000);
      else {
        logMessage('Failed to load Google Images after retries');
        return imagesData;
      }
    }
  }

  let lastCount = 0;
  let scrollAttempts = 0;
  const maxScrollAttempts = 100;

  // 滾動頁面，加載更多圖像，直到達到 maxImages 或無法加載更多
  while (imagesData.length < maxImages && scrollAttempts < maxScrollAttempts) {
    try {
      const moreButton = await page.$('input[value="Show more"]');
      if (moreButton) {
        logMessage('Clicking "Show more" button');
        await moreButton.click();
        await page.waitForTimeout(5000); // 等待頁面加載，可能因網絡或反爬蟲限制導致圖像數量不足
      } else {
        await page.evaluate('window.scrollBy(0, 1000)');
        await page.waitForTimeout(5000); // 等待時間較長，確保圖像加載
      }

      const imgElements = await page.$$('img');
      logMessage(`Found ${imgElements.length} img elements on page`);
      for (const img of imgElements) {
        const src = (await img.getAttribute('data-src')) || (await img.getAttribute('src'));
        const alt = await img.getAttribute('alt');
        if (
          src &&
          (src.startsWith('http') || src.startsWith('https')) &&
          imagesData.length < maxImages &&
          !imagesData.some((imgData) => imgData.src === src)
        ) {
          if (alt) {
            // 篩選邏輯：檢查 alt 文本是否匹配 DOG_BREED_TERMS 或包含“dog”/“puppy”
            // 若過嚴（例如僅接受特定品種），可能導致大量有效圖像被排除
            const matchedBreed = DOG_BREED_TERMS.find((term) => alt.toLowerCase().includes(term)) ||
                                (alt.toLowerCase().includes('dog') || alt.toLowerCase().includes('puppy') ? 'generic_dog' : null);
            if (matchedBreed && (breedCounts[matchedBreed] || 0) < MAX_IMAGES_PER_BREED) {
              imagesData.push({ src, alt, filename: null, breed: matchedBreed });
              breedCounts[matchedBreed] = (breedCounts[matchedBreed] || 0) + 1;
              logMessage(`Accepted image: src=${src}, alt=${alt}, breed=${matchedBreed}`);
            } else {
              logMessage(`Skipped image due to alt or breed limit: src=${src}, alt=${alt}`);
            }
          } else {
            logMessage(`Skipped image due to missing alt: src=${src}`);
          }
        } else {
          logMessage(`Skipped image: src=${src || 'No src'}, alt=${alt || 'No alt text'}`);
        }
      }

      if (imagesData.length === lastCount && !moreButton) {
        logMessage('No more images to load');
        break;
      }
      lastCount = imagesData.length;
      scrollAttempts++;
    } catch (error: any) {
      logMessage(`Error during image scraping: ${error.message}`);
      break;
    }
  }

  logMessage(`Collected ${imagesData.length} image metadata entries for "${keyword}"`);
  return imagesData;
}

// Download image
// 下載圖像並檢查尺寸是否符合要求
// 參數：
// - src: 圖像 URL
// - filename: 保存路徑
// - retries: 重試次數（默認 5 次）
// 返回：是否成功下載
// 注意：若網絡不穩定或圖像尺寸過小（<100x100），可能減少最終圖像數量
async function downloadImage(src: string, filename: string, retries: number = 5): Promise<boolean> {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await axios.get(src, { responseType: 'arraybuffer', timeout: 10000 });
      if (response.status === 200) {
        const buffer = response.data;
        const metadata = await sharp(buffer).metadata();
        if (metadata.width < 100 || metadata.height < 100) {
          logMessage(`Skipped image ${src}: too small (${metadata.width}x${metadata.height})`);
          return false;
        }
        writeFileSync(filename, buffer);
        logMessage(`Downloaded image: ${filename}`);
        return true;
      }
      logMessage(`Failed to download image: ${src} (status: ${response.status})`);
      return false;
    } catch (err: any) {
      logMessage(`Download attempt ${i + 1} failed for ${src}: ${err.message}`);
    }
  }
  logMessage(`Download image ${src} failed after ${retries} retries`);
  return false;
}

// Process image
// 處理圖像，調整尺寸並壓縮到指定大小
// 參數：
// - inputPath: 原始圖像路徑
// - outputPath: 處理後圖像路徑
// 返回：是否成功處理
// 注意：若無法壓縮到 MAX_SIZE_KB（50KB），圖像會被跳過，可能減少最終數量
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

// Main function
// 主函數，負責協調圖像收集、處理和存儲
// 參數：
// - userId: 用戶 ID，來自 server.ts 的 /search 端點
// - keywords: 搜索關鍵字列表，來自 search.html 表單或默認值
// 用途：
// - 創建 searches 表記錄，生成 search_id
// - 調用 searchImages 收集圖像
// - 下載並處理圖像，存儲到 images 表
// 與 search.html 的連動：
// - 用戶在 search.html 提交關鍵字，發送到 server.ts 的 /search 端點
// - server.ts 調用 main(userId, keywords)，傳遞用戶 ID 和關鍵字
// 與 server.ts 的連動：
// - server.ts 的 /search 端點負責驗證用戶並調用 main
// - main 存儲的圖像數據通過 server.ts 的 /api/images 端點返回給 search.html 和 results.html
// 注意：若圖像數量不足，檢查 log.txt 中的“Skipped”記錄，可能是篩選過嚴或網絡問題
export async function main(userId: number | null, keywords: string[] = ['dog images']): Promise<void> {
  logMessage(`Starting image collection for user ${userId || 'anonymous'}, keywords: "${keywords.join(', ')}"`);
  const breedCounts: { [key: string]: number } = {};

  initDatabase(async (err, db) => {
    if (err) {
      logMessage(`Failed to initialize database: ${err.message}`);
      return;
    }

    try {
      // Insert search record and get searchId
      const searchId = await new Promise<number>((resolve, reject) => {
        db.run(
          'INSERT INTO searches (user_id, keyword, image_count) VALUES (?, ?, ?)',
          [userId, keywords.join(', '), 0],
          function (err) {
            if (err) {
              logMessage(`Insert search error: ${err.message}`);
              reject(err);
            } else {
              logMessage(`Created search record with ID: ${this.lastID}`);
              resolve(this.lastID);
            }
          }
        );
      });

      let imagesData: ImageData[] = [];
      // Try each keyword
      for (const kw of keywords) {
        if (imagesData.length >= MAX_IMAGES) break;
        const browser = await chromium.launch({ headless: true });
        try {
          const page = await browser.newPage();
          const newImages = await searchImages(page, kw, MAX_IMAGES - imagesData.length, breedCounts);
          imagesData = imagesData.concat(newImages);
          logMessage(`Total images after keyword "${kw}": ${imagesData.length}`);
        } catch (error: any) {
          logMessage(`Error processing keyword "${kw}": ${error.message}`);
        } finally {
          await browser.close();
        }
      }

      if (imagesData.length < MIN_IMAGES) {
        logMessage(`Collected ${imagesData.length} images, less than required ${MIN_IMAGES}`);
      }

      // Log breed distribution
      // 記錄各品種的圖像數量，便於檢查數據集多樣性
      logMessage(`Image distribution by breed: ${JSON.stringify(breedCounts, null, 2)}`);

      let processedCount = 0;
      await new Promise<void>((resolve, reject) => {
        db.run('BEGIN TRANSACTION', async (err) => {
          if (err) {
            logMessage(`Transaction start error: ${err.message}`);
            reject(err);
            return;
          }

          const downloadPromises = imagesData.slice(0, MAX_IMAGES).map(async (image, index) => {
            const filename = `dog_${searchId}_${index + 1}.jpg`;
            const filePath = join(OUTPUT_DIR, filename);
            const tempFilename = join(OUTPUT_DIR, `temp_${searchId}_${index + 1}.jpg`);
            try {
              if (await downloadImage(image.src, tempFilename)) {
                if (await processImage(tempFilename, filePath)) {
                  return new Promise((resolve) => {
                    db.run(
                      'INSERT INTO images (src, alt, filename, user_id, search_id, is_relevant) VALUES (?, ?, ?, ?, ?, ?)',
                      [`/dog_images/${filename}`, image.alt, filename, userId, searchId, null],
                      (err) => {
                        if (err) {
                          logMessage(`Insert image error: ${err.message}`);
                          resolve(null);
                        } else {
                          processedCount++;
                          resolve(image);
                        }
                      }
                    );
                  });
                }
              }
              return null;
            } finally {
              if (existsSync(tempFilename)) unlinkSync(tempFilename);
            }
          });

          try {
            await Promise.all(downloadPromises);
            await new Promise<void>((resolve, reject) => {
              db.run('UPDATE searches SET image_count = ? WHERE id = ?', [processedCount, searchId], (err) => {
                if (err) {
                  logMessage(`Update search image_count error: ${err.message}`);
                  reject(err);
                } else {
                  resolve();
                }
              });
            });

            await new Promise<void>((resolve, reject) => {
              db.run('COMMIT', (err) => {
                if (err) {
                  logMessage(`Transaction commit error: ${err.message}`);
                  db.run('ROLLBACK', () => reject(err));
                  return;
                }
                resolve();
              });
            });

            await new Promise<void>((resolve, reject) => {
              db.get('SELECT COUNT(*) as count FROM images WHERE search_id = ?', [searchId], (err, result: { count: number }) => {
                if (err) {
                  logMessage(`Count query error: ${err.message}`);
                  reject(err);
                } else {
                  logMessage(`Processed ${processedCount} images for search ${searchId}. Total in database: ${result.count}`);
                  resolve();
                }
              });
            });
            resolve();
          } catch (error: any) {
            logMessage(`Transaction error: ${error.message}`);
            db.run('ROLLBACK', () => reject(error));
          }
        });
      });
    } catch (err: any) {
      logMessage(`Main execution failed: ${err.message}`);
      db.run('ROLLBACK', () => {});
    } finally {
      db.close((err) => {
        if (err) logMessage(`Database close error: ${err.message}`);
        else logMessage('Database connection closed');
      });
    }
  });
}

// Run standalone
// 獨立運行時的入口，測試用默認關鍵字
// 注意：實際使用時，關鍵字由 search.html 通過 server.ts 傳入
if (require.main === module) {
  main(null, [
    'chihuahua', '吉娃娃',
    'pomeranian', '博美犬',
    'yorkshire terrier', '約克夏梗',
    'maltese', '馬爾濟斯犬',
    'dachshund', '臘腸犬',
    'miniature poodle', '貴賓犬',
    'french bulldog', '法國鬥牛犬',
    'bichon frise', '比熊犬',
    'shih tzu', '西施犬',
    'miniature schnauzer', '迷你雪納瑞'
  ]).catch((err: any) => {
    logMessage(`Main execution failed: ${err.message}`);
  });
}