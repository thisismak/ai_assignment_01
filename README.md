# 習作一：人工智能原理及應用 - 自動搜集圖像數據集與初步處理

## 概述
本程式實現了自動搜集“各種不同品種的狗”圖像數據集的功能，包括：
- 使用 Playwright 搜尋 Google 圖像，收集 3000-5000 個圖像的 `src` 和 `alt`。
- 下載圖像並使用 Sharp 處理為 500x500 像素 JPEG 格式，檔案大小不超過 50KB。
- 將圖像元數據儲存到 SQLite 數據庫。
- 完全自動化，無需人工干預。

## 設置與運行
1. 安裝依賴：
   ```bash
   npm install
   npx playwright install