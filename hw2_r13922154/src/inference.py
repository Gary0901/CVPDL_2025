from ultralytics import YOLO
import os
import glob
from PIL import Image

# --- 配置參數 ---
# 1. 您的權重檔案路徑
WEIGHTS_PATH = "my_yolo_experiments/experiments1/weights/best.pt" 
# 2. 包含所有測試圖片的資料夾路徑
# 假設您的測試圖片在 'test_images' 資料夾中
SOURCE_DIR = "../data/CVPDL_hw2/CVPDL_hw2/test/" 
# 3. 指定輸出資料夾路徑
# 例如：'submissions/' 或 '/path/to/my/results/'
OUTPUT_DIR = "submissions/" 

# 4. 輸出檔案名稱（僅檔名部分）
OUTPUT_FILENAME = "final.csv"

# 5. 輸出檔案名稱
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def generate_submission_csv():
    # 載入模型
    print(f"正在載入模型權重: {WEIGHTS_PATH}")
    try:
        model = YOLO(WEIGHTS_PATH)
    except FileNotFoundError:
        print(f"錯誤：找不到權重檔案於 {WEIGHTS_PATH}")
        print("請確認路徑是否正確。")
        return

    # 獲取所有圖片檔案的路徑
    # 支援常見的圖片格式
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_paths.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
        
    if not image_paths:
        print(f"錯誤：在資料夾 {SOURCE_DIR} 中找不到任何圖片。請確認路徑。")
        return
    
    # --- 新增步驟：檢查並創建輸出目錄 ---
    print(f"檢查並創建輸出目錄: {OUTPUT_DIR}")
    try:
        # exist_ok=True 表示如果目錄已存在，則不會拋出錯誤
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"錯誤：無法創建輸出目錄 {OUTPUT_DIR}。原因: {e}")
        return

    # 準備寫入 CSV 檔案
    print(f"將結果寫入 {OUTPUT_CSV_FILE}...")
    with open(OUTPUT_CSV_FILE, 'w') as f:
        # 寫入 CSV 標題
        f.write("Image_ID,PredictionString\n")

        # 逐一處理每張圖片
        # ⚠️ 由於 Image_ID 現在是數字，我們按數字排序以確保順序正確
        # 為了正確排序，我們需要一個 (數字ID, 檔案路徑) 的列表
        processed_paths = []
        for path in image_paths:
            base_name = os.path.splitext(os.path.basename(path))[0] # e.g., 'img0001'
            
            # --- 關鍵修改：提取純數字 ID ---
            # 假設格式總是 'img' + 數字
            try:
                # 移除 'img' 前綴並轉換為整數 (自動去除前導零)
                numeric_id = int(base_name.replace('img', ''))
                processed_paths.append((numeric_id, path))
            except ValueError:
                print(f"警告: 檔案名 {base_name} 格式不符合 'imgXXXX'，將跳過。")
                continue
        
        # 按數字 ID 排序
        processed_paths.sort(key=lambda x: x[0])


        for numeric_id, image_path in processed_paths:
            
            # 將 Image_ID 設置為純數字
            image_id = str(numeric_id) # e.g., '1', '2', ...
            
            # --- 進行推論 (不變) ---
            results = model.predict(
                source=image_path,
                conf=0.3,  #Confidence Threshold 
                iou=0.7,  #IoU Threshold for NMS
                imgsz=1920,
                device=1,
                verbose=False,
                augment=True # <--- 啟用 TTA！
            )
            
            # --- 處理推論結果並格式化 PredictionString (不變) ---
            prediction_string = []
            
            if results and results[0].boxes:
                boxes = results[0].boxes
                
                try:
                    img = Image.open(image_path)
                    original_h, original_w = img.height, img.width
                    img.close()
                except Exception as e:
                    print(f"無法讀取圖片 {image_path} 的大小：{e}。跳過。")
                    continue
                
                # 假設輸出格式為：conf xmin ymin width height class_index
                for box in boxes:
                    conf = box.conf.item()     
                    cls = int(box.cls.item())  
                    x1, y1, x2, y2 = box.xyxy[0].tolist() 
                    
                    x_min = x1
                    y_min = y1
                    width = x2 - x1
                    height = y2 - y1
                    
                    prediction_string.append(
                        f"{conf:.6f} {x_min:.2f} {y_min:.2f} {width:.2f} {height:.2f} {cls}"
                    )

                final_prediction = " ".join(prediction_string)
            
            else:
                final_prediction = ""
            
            # 寫入 CSV
            f.write(f"{image_id},{final_prediction}\n") # 確保 Image_ID 是純數字

    print(f"\n成功生成 {OUTPUT_CSV_FILE}。共處理 {len(processed_paths)} 張圖片。")

if __name__ == '__main__':
    # ⚠️ 在執行前，請務必確認 WEIGHTS_PATH 和 SOURCE_DIR 正確！
    generate_submission_csv()