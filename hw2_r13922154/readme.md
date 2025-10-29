## HW2：基於 YOLO11x 的長尾物件偵測 (Long-Tailed Object Detection)
本專案實作了使用 YOLO11x 進行長尾物件偵測的完整流程。我們採用了客製化的 $\text{VFL}$ 類別感知重加權 (Class-Aware Reweighting) 策略，結合 高解析度輸入 和 數據增強 ($\text{MixUp}$)，以提升最稀有類別 ($\text{hov}, \text{person}$) 的 $\text{Recall}$ 和整體 $\text{mAP50-95}$ 分數.

```
專案結構 (Project Structure)
/hw2_r13922154
├── report_r13922154.pdf        # 報告 (含長尾 VFL 數學推導與分析)
├── code_r13922154.zip
│   ├── src/
│   │   ├── convert_to_yolo.py     # 1. 原始標註格式轉換為 YOLO 格式 
│   │   ├── split_dataset.py       # 2. 圖片/標籤分割為 train/val 子集 
│   │   ├── train.py               # 3. 模型訓練腳本 
│   │   ├── inference.py           # 4. 推論腳本 (含優化後的 conf/iou 參數)
|   |   ├── hw2_dataset.yaml       # 5. 數據集配置檔
│   |   └── my_yolo_experiments/   # 6. 訓練結果儲存目錄
│   ├── requirements.txt       # 必要的 Python 套件清單
│   ├── data/                  # 存放資料之位置
│   └── readme.md              # 本文件
```

### 1. 環境設定 (Environment Setup)
本專案建議使用 $\text{Python} \ge 3.8$ 的虛擬環境，並需確保 $\text{PyTorch}$ 版本與 $\text{GPU CUDA}$ 版本兼容。

#### 1.1 創建虛擬環境
```
python3 -m venv venv
# 啟用虛擬環境
source venv/bin/activate
```

#### 1.2 安裝依賴套件
本專案所需的套件已列於 requirements.txt 中。
```
pip install -r requirements.txt
```
---

### 2. 數據準備 (Data Preparation)
請將 $\text{Kaggle}$ 下載的原始訓練資料放置於專案根目錄下的 $\mathbf{data/}$ 資料夾中。
結構為 : `/data/CVPDL_hw2/CVPDL_hw2/train or test`
該資料夾應同時包含圖片檔案（例如 $\text{img0001.png}$）和對應的標註檔案（例如 $\text{img0001.txt}$）。

#### 2.1 轉換標註格式 (Kaggle Format to YOLO)
原始標註檔案格式為 $\text{<class label>,<Top-left X>,<Top-left Y>,<Bounding box width>,<Bounding box height>}$。我們需要將其轉換為 $\text{YOLO}$ 所需的正規化格式 $\text{<class> <x\_center> <y\_center> <width> <height>}$。
執行腳本：
```
# 執行轉換腳本
# 輸出路徑: ../data/train_yolo_labels
python3 src/convert_to_yolo.py
```

#### 2.2 分割訓練/驗證集
將轉換後的 $\text{YOLO}$ 格式標籤和圖片，以 $\mathbf{80\%}$ 訓練集 / $\mathbf{20\%}$ 驗證集的比例進行分割和複製。

```
# 執行分割腳本
# 讀取來源: 圖片在 ../data/CVPDL_hw2/CVPDL_hw2/train，標籤在 ../data/train_yolo_labels
# 輸出目錄: ../data/datasets (包含 images/train, labels/train, images/val, labels/val)
python3 src/split_dataset.py
``` 

#### 2.3 創建 YOLO 數據配置檔 (hw2_dataset.yaml)
如無$\mathbf{hw2\_dataset.yaml}$，請手動創建 $\mathbf{hw2\_dataset.yaml}$ 檔案於src/下，用於指定數據集的路徑和類別資訊：

```
# 告訴 YOLO 你的資料集根目錄在哪裡
# '.' 表示 'hw2_dataset.yaml' 所在的當前資料夾
path: ../data/datasets

# 相對於 path 的路徑
train: images/train  # 訓練圖片資料夾
val: images/val      # 驗證圖片資料夾

# 測試集圖片路徑 (用於 model.predict())
test: ../test_images  # 注意：我們把測試集放在 datasets 外面

# 類別 (Classes)
nc: 4  # 類別總數

# 類別名稱 (必須按照索引 0, 1, 2, 3 的順序)
# 根據簡報 P.11 [cite: 127] 和 P.2 [cite: 17]
names:
  0: car
  1: hov
  2: person
  3: motorcycle
```

### 3. 模型訓練 (Model Training)
#### 3. 執行訓練
使用 train.py 開始訓練，訓練結果將儲存於 my_yolo_experiments/ 目錄下。

```
python3 src/train.py \
```

### 4. 預測與提交 (Inference and Submission)
使用訓練完成後最佳的權重檔案 (my_yolo_experiments/.../weights/best.pt) 進行測試集推論。

```
python3 src/inference.py \
``` 
產生的 submission_final.csv 即可用於 Kaggle 提交。
