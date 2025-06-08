# 習作一：自動搜集圖像數據集與初步處理

## 學生資料
- 姓名：麥志榮
- 學號：23816955
- 主題編號：5
- 主題名稱：各種不同品種的狗

# 程式運行記錄
- 安裝流程及執行流程
1. 安裝python 3.10
https://www.python.org/downloads/release/python-3100/
2. 升級 pip 到最新版本
python -m pip install --upgrade pip
3. 安裝依賴
python -m pip install pillow tensorflow imagehash numpy requests playwright
playwright install
4. TensorFlow 2.16.1 是已知與 Windows 上的 Python 3.10 兼容的穩定版本。
pip uninstall tensorflow
pip install tensorflow==2.16.1
pip show tensorflow
5. 運行程式
python script.py
- 運作日誌
  -  breed_distribution.csv
breed,image_count
Maltese,400
Yorkshire Terrier,400
Pomeranian,400
Chihuahua,400
Miniature Schnauzer,400
Shih Tzu,400
Poodle,400
Dachshund,400
Shiba Inu,400
Labrador Retriever,400

  - image_failure_report.csv
keyword,url,alt_text,failure_stage,failure_reason
Poodle,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Shiba Inu,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Shiba Inu,https://r.bing.com/rp/UYtUYDcn1oZlFG-YfBPz59zejYI.svg,拖放圖片到這裡,processing,Compression or processing error
Shiba Inu,https://r.bing.com/rp/KC_nX2_tPPyFvVw1RK20Yu1FyDk.svg,貼上圖片或網址,processing,Compression or processing error
Dachshund,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Miniature Schnauzer,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Miniature Schnauzer,https://r.bing.com/rp/UYtUYDcn1oZlFG-YfBPz59zejYI.svg,拖放圖片到這裡,processing,Compression or processing error
Miniature Schnauzer,https://r.bing.com/rp/KC_nX2_tPPyFvVw1RK20Yu1FyDk.svg,貼上圖片或網址,processing,Compression or processing error
Miniature Schnauzer,https://r.bing.com/rp/hx-eea1zqtCz4K0bW2uH_oN7Fs4.jpg,太陽眼鏡,database,Duplicate URL
Miniature Schnauzer,https://r.bing.com/rp/ln5TQq6AIWfcBlduDk-5bnaJMpY.jpg,雪梨歌劇院,database,Duplicate URL
Miniature Schnauzer,https://r.bing.com/rp/cfeVf2-uV0hUo3ToTbLjztuomWk.jpg,盧浮宮,database,Duplicate URL
Miniature Schnauzer,https://r.bing.com/rp/lvCKZ07bEYtoYmY62ifMzVa0RIE.jpg,兩隻狗,database,Duplicate URL
Miniature Schnauzer,https://r.bing.com/rp/ni3MyKKVu9pK0SgY6gb6Z2NOGpg.jpg,手工海草,database,Duplicate URL
Miniature Schnauzer,https://r.bing.com/rp/-A5v-hTPFRzEXEMXLO7124F8nt0.svg,GIF 自動播放設定,processing,Compression or processing error
Miniature Schnauzer,https://r.bing.com/rp/Q5BJPjebyYN5QiqznkcMQmLrF9U.svg,關閉對話方塊,processing,Compression or processing error
Chihuahua,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Yorkshire Terrier,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Yorkshire Terrier,https://r.bing.com/rp/UYtUYDcn1oZlFG-YfBPz59zejYI.svg,拖放圖片到這裡,processing,Compression or processing error
Yorkshire Terrier,https://r.bing.com/rp/KC_nX2_tPPyFvVw1RK20Yu1FyDk.svg,貼上圖片或網址,processing,Compression or processing error
Yorkshire Terrier,https://r.bing.com/rp/hx-eea1zqtCz4K0bW2uH_oN7Fs4.jpg,太陽眼鏡,database,Duplicate URL
Yorkshire Terrier,https://r.bing.com/rp/ln5TQq6AIWfcBlduDk-5bnaJMpY.jpg,雪梨歌劇院,database,Duplicate URL
Yorkshire Terrier,https://r.bing.com/rp/cfeVf2-uV0hUo3ToTbLjztuomWk.jpg,盧浮宮,database,Duplicate URL
Yorkshire Terrier,https://r.bing.com/rp/lvCKZ07bEYtoYmY62ifMzVa0RIE.jpg,兩隻狗,database,Duplicate URL
Yorkshire Terrier,https://r.bing.com/rp/ni3MyKKVu9pK0SgY6gb6Z2NOGpg.jpg,手工海草,database,Duplicate URL
Yorkshire Terrier,https://r.bing.com/rp/-A5v-hTPFRzEXEMXLO7124F8nt0.svg,GIF 自動播放設定,processing,Compression or processing error
Yorkshire Terrier,https://r.bing.com/rp/Q5BJPjebyYN5QiqznkcMQmLrF9U.svg,關閉對話方塊,processing,Compression or processing error
Pomeranian,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Pomeranian,https://r.bing.com/rp/UYtUYDcn1oZlFG-YfBPz59zejYI.svg,拖放圖片到這裡,processing,Compression or processing error
Pomeranian,https://r.bing.com/rp/KC_nX2_tPPyFvVw1RK20Yu1FyDk.svg,貼上圖片或網址,processing,Compression or processing error
Pomeranian,https://r.bing.com/rp/hx-eea1zqtCz4K0bW2uH_oN7Fs4.jpg,太陽眼鏡,database,Duplicate URL
Pomeranian,https://r.bing.com/rp/ln5TQq6AIWfcBlduDk-5bnaJMpY.jpg,雪梨歌劇院,database,Duplicate URL
Pomeranian,https://r.bing.com/rp/cfeVf2-uV0hUo3ToTbLjztuomWk.jpg,盧浮宮,database,Duplicate URL
Pomeranian,https://r.bing.com/rp/lvCKZ07bEYtoYmY62ifMzVa0RIE.jpg,兩隻狗,database,Duplicate URL
Pomeranian,https://r.bing.com/rp/ni3MyKKVu9pK0SgY6gb6Z2NOGpg.jpg,手工海草,database,Duplicate URL
Pomeranian,https://r.bing.com/rp/-A5v-hTPFRzEXEMXLO7124F8nt0.svg,GIF 自動播放設定,processing,Compression or processing error
Pomeranian,https://r.bing.com/rp/Q5BJPjebyYN5QiqznkcMQmLrF9U.svg,關閉對話方塊,processing,Compression or processing error
Maltese,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Maltese,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTY_G64jeJMDFue_woAcx4H0EifwWTZTVcRew&s,Maltese vs Poodle: A Comprehensive Guide to Choosing Pet,database,Duplicate URL
Maltese,https://r.bing.com/rp/UYtUYDcn1oZlFG-YfBPz59zejYI.svg,拖放圖片到這裡,processing,Compression or processing error
Maltese,https://r.bing.com/rp/KC_nX2_tPPyFvVw1RK20Yu1FyDk.svg,貼上圖片或網址,processing,Compression or processing error
Maltese,https://r.bing.com/rp/hx-eea1zqtCz4K0bW2uH_oN7Fs4.jpg,太陽眼鏡,database,Duplicate URL
Maltese,https://r.bing.com/rp/ln5TQq6AIWfcBlduDk-5bnaJMpY.jpg,雪梨歌劇院,database,Duplicate URL
Maltese,https://r.bing.com/rp/cfeVf2-uV0hUo3ToTbLjztuomWk.jpg,盧浮宮,database,Duplicate URL
Maltese,https://r.bing.com/rp/lvCKZ07bEYtoYmY62ifMzVa0RIE.jpg,兩隻狗,database,Duplicate URL
Maltese,https://r.bing.com/rp/ni3MyKKVu9pK0SgY6gb6Z2NOGpg.jpg,手工海草,database,Duplicate URL
Maltese,https://r.bing.com/rp/-A5v-hTPFRzEXEMXLO7124F8nt0.svg,GIF 自動播放設定,processing,Compression or processing error
Maltese,https://r.bing.com/rp/Q5BJPjebyYN5QiqznkcMQmLrF9U.svg,關閉對話方塊,processing,Compression or processing error
Shih Tzu,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Shih Tzu,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQniEVHmU9uLKpoecgWtV20XDcNpXUy3R86Hw&s,Grooming Your Maltese Shih tzu Puppies: Best Practices and Tips - Dog love  Services,database,Duplicate URL
Shih Tzu,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSFVrXURc7Y0ZGYbkMhWIZOTzDQWBEAbhaz-w&s,"Maltese Shih Tzu breed insights: care, personality & expert tips | Lyka Blog",database,Duplicate URL
Shih Tzu,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVQ0KmkUQUcdtot6WlijmjyudZJVqihRAoCg&s,Maltese Shih Tzu Mix Breed Guide: Characteristics & Facts | Bored Panda,database,Duplicate URL
Shih Tzu,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSHYTIi1jTmNStAig66jdwIUQTluIRUfAN9A&s,The Maltese Shih Tzu (Malshi): Best Dog Ever! - PetHelpful,database,Duplicate URL
Shih Tzu,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTsCculSlflS5O9eafE4aYIYnrW98G4aSP8MA&s,Maltese Shih Tzu Dog Breed Information | Temperament & Health,database,Duplicate URL
Shih Tzu,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_ILr0WWqkqd0usWjlUqKqtxLBO7FVG_3Z2Q&s,Maltese Shih Tzu Dog Breed Information | Temperament & Health,database,Duplicate URL
Shih Tzu,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIwsjfm68sVvsRPxE9Svo50PsxrHTWZo_PzA&s,Maltese Shih Tzu Dog Breed Information | Temperament & Health,database,Duplicate URL
Shih Tzu,https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTg-HFUxQ_gQAT9qg6ObviVvUgv10LFW53iPg&s,Maltese Shih Tzu Mix: The Affectionate and Playful Malshi,database,Duplicate URL
Shih Tzu,https://r.bing.com/rp/UYtUYDcn1oZlFG-YfBPz59zejYI.svg,拖放圖片到這裡,processing,Compression or processing error
Shih Tzu,https://r.bing.com/rp/KC_nX2_tPPyFvVw1RK20Yu1FyDk.svg,貼上圖片或網址,processing,Compression or processing error
Shih Tzu,https://r.bing.com/rp/hx-eea1zqtCz4K0bW2uH_oN7Fs4.jpg,太陽眼鏡,database,Duplicate URL
Shih Tzu,https://r.bing.com/rp/ln5TQq6AIWfcBlduDk-5bnaJMpY.jpg,雪梨歌劇院,database,Duplicate URL
Shih Tzu,https://r.bing.com/rp/cfeVf2-uV0hUo3ToTbLjztuomWk.jpg,盧浮宮,database,Duplicate URL
Shih Tzu,https://r.bing.com/rp/lvCKZ07bEYtoYmY62ifMzVa0RIE.jpg,兩隻狗,database,Duplicate URL
Shih Tzu,https://r.bing.com/rp/ni3MyKKVu9pK0SgY6gb6Z2NOGpg.jpg,手工海草,database,Duplicate URL
Labrador Retriever,https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg,Google,processing,Compression or processing error
Labrador Retriever,https://r.bing.com/rp/UYtUYDcn1oZlFG-YfBPz59zejYI.svg,拖放圖片到這裡,processing,Compression or processing error
Labrador Retriever,https://r.bing.com/rp/KC_nX2_tPPyFvVw1RK20Yu1FyDk.svg,貼上圖片或網址,processing,Compression or processing error
Labrador Retriever,https://r.bing.com/rp/hx-eea1zqtCz4K0bW2uH_oN7Fs4.jpg,太陽眼鏡,database,Duplicate URL
Labrador Retriever,https://r.bing.com/rp/ln5TQq6AIWfcBlduDk-5bnaJMpY.jpg,雪梨歌劇院,database,Duplicate URL
Labrador Retriever,https://r.bing.com/rp/cfeVf2-uV0hUo3ToTbLjztuomWk.jpg,盧浮宮,database,Duplicate URL
Labrador Retriever,https://r.bing.com/rp/lvCKZ07bEYtoYmY62ifMzVa0RIE.jpg,兩隻狗,database,Duplicate URL
Labrador Retriever,https://r.bing.com/rp/ni3MyKKVu9pK0SgY6gb6Z2NOGpg.jpg,手工海草,database,Duplicate URL


  - image_collection.log
2025-06-08 15:27:52,187 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_391_3991.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,245 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_392_3992.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,261 - ERROR - Error processing dog_images\Labrador_Retriever\temp_393_3993.jpg: cannot identify image file 'dog_images\\Labrador_Retriever\\temp_393_3993.jpg'
2025-06-08 15:27:52,292 - ERROR - Error processing dog_images\Labrador_Retriever\temp_393_3993.jpg: cannot identify image file 'dog_images\\Labrador_Retriever\\temp_393_3993.jpg'
2025-06-08 15:27:52,575 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_393_3993.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,641 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_394_3994.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,669 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_395_3995.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,727 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_396_3996.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,797 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_397_3997.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,885 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_398_3998.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,937 - INFO - Successfully processed image: dog_images\Labrador_Retriever\Labrador_Retriever_399_3999.jpg for breed: Labrador Retriever
2025-06-08 15:27:52,953 - INFO - Completed keyword Labrador Retriever, total images: 4000, breed Labrador Retriever: 400
2025-06-08 15:27:52,964 - INFO - Database verification: 4000 images stored
2025-06-08 15:27:52,964 - INFO -  ` @     óB z   Ϥ  ơG4000
2025-06-08 15:27:52,964 - INFO -  ] L  alt 奻 L o     Ϥ  G86834
2025-06-08 15:27:52,964 - INFO -  U     Ѫ  Ϥ  G0
2025-06-08 15:27:52,964 - INFO -  B z   Ѫ  Ϥ  G32
2025-06-08 15:27:52,964 - INFO -     URL   L   Ϥ  G39
2025-06-08 15:27:52,964 - INFO -  ~ ؤ  G G{'Maltese': 400, 'Yorkshire Terrier': 400, 'Pomeranian': 400, 'Chihuahua': 400, 'Miniature Schnauzer': 400, 'Shih Tzu': 400, 'Poodle': 400, 'Dachshund': 400, 'Shiba Inu': 400, 'Labrador Retriever': 400}
2025-06-08 15:27:52,964 - INFO - Exported failure report to image_failure_report.csv
2025-06-08 15:27:52,964 - INFO - Exported breed distribution to breed_distribution.csv

## 功能完成度自評

（依照評分標準勾選完成的項目）

### 搜索和收集（30 分）

- 關鍵字搜索和結果相關性（15 分）

  - [X] 有效使用關鍵字，獲得相關圖像

- 收集圖像的網址和替代文字（15 分）
  - [X] 正確收集並記錄圖像的 src 或者下載頁面的 url 信息
  - [X] 正確收集並記錄圖像的 alt 信息

### 圖像下載和處理（50 分）

- 下載和儲檔（20 分）

  - [X] 成功下載 3000 至 5000 個圖像
  - [X] 圖像儲存在相應的文件夾
  - [X] 列出收集的圖像數量

- 調整和裁剪（20 分）

  - [X] 正確調整圖像大小為不超過 500x500 像素
  - [X] 置中裁剪

- 重新編碼和尺寸控制（10 分）
  - [X] 圖像重新編碼為 JPEG 格式
  - [X] 圖像質量設置在 50-80 之間
  - [X] 確保圖像大小不超過 50KB

### 自動化和代碼質量（20 分）

- 自動化程度（10 分）

  - [X] 實現數據搜集的自動化
  - [X] 實現圖像下載的自動化
  - [X] 實現圖像處理的自動化

- 程式碼結構和註釋（10 分）
  - [X] 代碼具有良好的模組化設計
  - [X] 代碼結構清晰易讀
  - [X] 註釋完整且有意義