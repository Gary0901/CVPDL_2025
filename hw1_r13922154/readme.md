## HW1：基於 YOLOv11x 的群養豬隻物體偵測 (Object Detection)
本專案實作了使用 YOLOv11x 進行群養豬隻物體偵測的完整流程。我們採用了客製化的 Partial Weight Loading 策略和針對 密集目標 的超參數調整，以最大化 mAP 
50:95分數。

專案結構 (Project Structure)
```
/hw1_r13922154
├── report_r13922154.pdf
├── code_r13922154zip
│   ├── src/
│   │   ├── conver_to_yolo.py  # 1. 轉換標註格式
│   │   ├── split_dataset.py   # 2. 分割訓練/驗證集
│   │   ├── train.py           # 3. 模型訓練腳本 (含 Partial Weight Loading)
│   │   └── inference.py       # 4. 推論腳本 (含 TTA)
│   ├── requirements.txt       # 必要的 Python 套件清單
│   └── readme.md              # 本文件
├── yolo_dataset/              # 訓練/驗證數據集 (由 split_dataset.py 產生)
└── runs/                      # 訓練結果儲存目錄 (由 train.py 產生)
```

### 1. 環境設定 (Environment Setup)
本專案建議使用 Python 虛擬環境 (Virtual Environment) 以確保套件版本一致性。

#### 1.1 創建虛擬環境
```
確保 Python 版本 >= 3.10
python3 -m venv venv
# 啟用虛擬環境
source venv/bin/activate
``` 

#### 1.2 安裝依賴套件
本專案所需的套件已列於 requirements.txt 中。

```
pip install -r requirements.txt
# 注意：你需要確保 PyTorch 版本與你的 GPU CUDA 版本兼容。
```

### 2. 數據準備 (Data Preparation)
請將 Kaggle 下載的數據集放置於專案根目錄下的 data/ 資料夾，結構應如下：data/ntu-cvpdl-2025-hw-1/train/img 和 data/ntu-cvpdl-2025-hw-1/train/gt.txt。

####　2.1 轉換標註格式 (Pascal VOC to YOLO)
原始數據集使用 <bb_left>, <bb_top>, <bb_width>, <bb_height> 格式。我們需要將其轉換為 YOLO 所需的正規化格式。

```
python3 src/conver_to_yolo.py \
    --gt-file "data/ntu-cvpdl-2025-hw-1/train/gt.txt" \
    --image-dir "data/ntu-cvpdl-2025-hw-1/train/img" \
    --output-labels-dir "yolo_labels"
# 轉換後的標註檔案將儲存於 yolo_labels/
```

### 2.2 分割訓練/驗證集
將圖片和對應的 YOLO 標籤分割成 train/ 和 val/ 兩個子集。

```
python3 src/split_dataset.py \
    --source-image-dir "data/ntu-cvpdl-2025-hw-1/train/img" \
    --source-label-dir "yolo_labels" \
    --target-root-dir "yolo_dataset" \
    --split-ratio 0.20
# 輸出目錄：yolo_dataset/images/train, yolo_dataset/labels/train, ...
```

#### 2.3 創建 YOLO 數據配置檔 (pigs.yaml)
請手動創建 pigs.yaml 檔案，用於指定數據集的路徑和類別資訊：

```
# Path:
path: ./yolo_dataset  # 數據集的根目錄
train: images/train  # 訓練圖片路徑
val: images/val      # 驗證圖片路徑

# Classes
nc: 1  # 只有一個類別 (pig)
names: ['pig']
```

### 3. 模型訓練 (Model Training)
#### 3.1 準備權重和模型配置
重要提醒： 請確保已下載yolo11x.pt (預訓練權重檔)，並將它們放置於專案根目錄。您可以從 [YOLO 官方資源] 或 [特定的模型庫] 獲取這些檔案。
將 yolo11x.yaml (模型結構) 和 yolo11x.pt (預訓練權重) 放置於專案根目錄。

train.py 內建了 Box Loss Weight = 10.0 和 NMS IoU = 0.6 的客製化超參數。

#### 3.2 執行訓練
使用 train.py 開始訓練，訓練結果將儲存於 runs/yolo11/ 目錄下。

```
python3 src/train.py \
    --data-yaml "pigs.yaml" \
    --pretrained-pt "yolo11x.pt" \
    --device "0" \
    --experiment-name "yolo11x_final_run"
```

### 4. 預測與提交 (Inference and Submission)
使用訓練完成後最佳的權重檔案 (runs/.../weights/best.pt) 進行測試集推論。

```
python3 src/inference.py \
    --model-path "runs/yolo11/yolo11x_final_run/weights/best.pt" \
    --test-image-dir "data/ntu-cvpdl-2025-hw-1/test/img" \
    --output-csv "submission_final.csv" \
    --device "0"
# 產生的 submission_final.csv 即可用於 Kaggle 提交。
```
