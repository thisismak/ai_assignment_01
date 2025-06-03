import { chromium, Page } from '@playwright/test';
import axios from 'axios';
import sharp from 'sharp';
import sqlite3 from 'sqlite3';
import { existsSync, mkdirSync, unlinkSync, writeFileSync, appendFileSync } from 'fs';
import { join } from 'path';

// Configuration
const OUTPUT_DIR = 'dog_images';
const DB_PATH = 'db.sqlite3';
const LOG_PATH = 'log.txt';
const MAX_IMAGES = 5000;
const MIN_IMAGES = 3000; // Lowered for testing
const MAX_SIZE_KB = 50;
const TARGET_SIZE = { width: 500, height: 500 };
const QUALITY_RANGE = { min: 50, max: 80 };
const KEYWORDS = ['specific dog breeds photos', 'purebred dog images', 'different dog breeds portraits', 'dogs', 'dog photos'];
const DOG_RELATED_TERMS = ['dog', 'breed', 'puppy', 'canine', 'labrador', 'poodle', 'bulldog'];
const USER_AGENTS = [
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15',
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0',
];

// Log function (Hong Kong time)
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
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  logMessage(`Created output directory: ${OUTPUT_DIR}`);
}

// Image data interface
interface ImageData {
  src: string;
  alt: string | null;
  filename: string | null;
}

// Initialize database
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
async function searchImages(page: Page, keyword: string, maxImages: number): Promise<ImageData[]> {
  logMessage(`Starting image search for keyword: "${keyword}"`);
  const imagesData: ImageData[] = [];

  // Rotate User-Agent
  const userAgent = USER_AGENTS[Math.floor(Math.random() * USER_AGENTS.length)];
  await page.setExtraHTTPHeaders({ 'user-agent': userAgent });
  await page.setViewportSize({ width: 1280, height: 720 });

  let retries = 3;
  while (retries > 0) {
    try {
      await page.goto(`https://www.google.com/search?q=${encodeURIComponent(keyword)}&tbm=isch`, {
        waitUntil: 'domcontentloaded',
        timeout: 30000,
      });
      logMessage('Successfully loaded Google Images page');
      await page.screenshot({ path: `google_images_screenshot_${keyword}}.jpg` });
      break;
    } catch (error: any) {
      logMessage(`Failed to load Google Images page (Retry ${4 - retries}): ${error.message}`);
      retries--;
      if (retries > 0) await page.waitForTimeout(5000);
    }
  }

  if (retries === 0) {
    logMessage('Failed to load Google Images after retries');
    return imagesData;
  }

  let lastCount = 0;
  let scrollAttempts = 0;
  const maxScrollAttempts = 50;

  while (imagesData.length < maxImages && scrollAttempts < maxScrollAttempts) {
    const moreButton = await page.$('input[value="Show more"]');
    if (moreButton) {
      logMessage('Clicking "Show more" button');
      await moreButton.click();
      await page.waitForTimeout(2000); // Reduced delay
    } else {
      await page.evaluate('window.scrollBy(0, 1000)');
      await page.waitForTimeout(2000);
    }

    const imgElements = await page.$$('img');
    logMessage(`Found ${imgElements.length} img elements on page`);
    for (const img of imgElements) {
      const src = (await img.getAttribute('data-src')) || (await img.getAttribute('src'));
      const alt = await img.getAttribute('alt');
      if (
        src &&
        (src.startsWith('http') || src.startsWith('https')) &&
        imagesData.length < maxImages
      ) {
        // Relaxed alt text filter
        if (!alt || DOG_RELATED_TERMS.some((term) => alt.toLowerCase().includes(term))) {
          imagesData.push({ src, alt: alt || null, filename: null });
          logMessage(`Accepted image: src=${src}, alt=${alt}`);
        } else {
          logMessage(`Skipped image due to alt: src=${src}, alt=${alt}`);
        }
      } else {
        logMessage(`Skipped image: src=${src}, alt=${alt}`);
      }
    }

    if (imagesData.length === lastCount && !moreButton) {
      logMessage('No more images to load');
      break;
    }
    lastCount = imagesData.length;
    scrollAttempts++;
  }

  logMessage(`Collected ${imagesData.length} image metadata entries for "${keyword}"`);
  return imagesData;
}

// Download image
async function downloadImage(src: string, filename: string, retries: number = 3): Promise<boolean> {
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
export async function main(userId: number | null, keyword: string = KEYWORDS[0]): Promise<void> {
  logMessage(`Starting image collection for user ${userId || 'anonymous'}, keyword: "${keyword}"`);
  initDatabase(async (err, db) => {
    if (err) {
      logMessage(`Failed to initialize database: ${err.message}`);
      return;
    }

    try {
      // Insert search record
      db.run(
        'INSERT INTO searches (user_id, keyword, image_count) VALUES (?, ?, ?)',
        [userId, keyword, 0],
        async function (err) {
          if (err) {
            logMessage(`Insert search error: ${err.message}`);
            db.close();
            return;
          }
          const searchId = this.lastID;
          logMessage(`Created search record with ID: ${searchId}`);

          let imagesData: ImageData[] = [];
          // Try each keyword until enough images are collected
          for (const kw of KEYWORDS) {
            if (imagesData.length >= MAX_IMAGES) break;
            const browser = await chromium.launch({ headless: true });
            const page = await browser.newPage();
            const newImages = await searchImages(page, kw, MAX_IMAGES - imagesData.length);
            imagesData = imagesData.concat(newImages);
            await browser.close();
            logMessage(`Total images after keyword "${kw}": ${imagesData.length}`);
          }

          if (imagesData.length < MIN_IMAGES) {
            logMessage(`Collected ${imagesData.length} images, less than required ${MIN_IMAGES}`);
          }

          let processedCount = 0;
          db.run('BEGIN TRANSACTION', async (err) => {
            if (err) {
              logMessage(`Transaction start error: ${err.message}`);
              db.close();
              return;
            }

            const downloadPromises = imagesData.slice(0, MAX_IMAGES).map(async (image, index) => {
              const filename = `dog_${searchId}_${index + 1}.jpg`;
              const filePath = join(OUTPUT_DIR, filename);
              const tempFilename = join(OUTPUT_DIR, `temp_${searchId}_${index + 1}.jpg`);
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
                        if (existsSync(tempFilename)) unlinkSync(tempFilename);
                      }
                    );
                  });
                }
                if (existsSync(tempFilename)) unlinkSync(tempFilename);
              }
              return null;
            });

            await Promise.all(downloadPromises);
            db.run('UPDATE searches SET image_count = ? WHERE id = ?', [processedCount, searchId], (err) => {
              if (err) logMessage(`Update search image_count error: ${err.message}`);
            });

            db.run('COMMIT', (err) => {
              if (err) {
                logMessage(`Transaction commit error: ${err.message}`);
                db.run('ROLLBACK', () => db.close());
                return;
              }
              db.get('SELECT COUNT(*) as count FROM images WHERE search_id = ?', [searchId], (err, result: { count: number }) => {
                if (err) {
                  logMessage(`Count query error: ${err.message}`);
                  db.close();
                  return;
                }
                logMessage(`Processed ${processedCount} images for search ${searchId}. Total in database: ${result.count}`);
                db.close();
              });
            });
          });
        }
      );
    } catch (err: any) {
      logMessage(`Main execution failed: ${err.message}`);
      db.run('ROLLBACK', () => db.close());
    }
  });
}

// Run standalone
if (require.main === module) {
  main(null).catch((err: any) => {
    logMessage(`Main execution failed: ${err.message}`);
  });
}