## 學生資料
- 姓名：麥志榮
- 學號：23816955
- 主題編號：5
- 主題名稱：各種不同品種的狗

# 安裝指南
1. 安裝python 3.10
https://www.python.org/downloads/release/python-3100/
2. 升級 pip 到最新版本
python -m pip install --upgrade pip
3. 安裝依賴
python -m pip install pillow tensorflow imagehash numpy requests playwright
playwright install
4. TensorFlow 2.16.1 is a stable version known to work with Python 3.10 on Windows.
pip uninstall tensorflow
pip install tensorflow==2.16.1
pip show tensorflow
5. 運行程式
python script.py