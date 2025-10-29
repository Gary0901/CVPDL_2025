import os
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import warnings
import torch
import gc 

# 忽略不重要的 PyTorch/Ultralytics 警告
# 修正：刪除不正確的語法 'warnings.filter sparingly'
warnings.filterwarnings("ignore", category=UserWarning)

# --- 配置區 (已根據 imgsz=800 訓練更新) ---

# 1. 推論超參數
CONFIDENCE_THRESHOLD = 0.01  # 保持極低
IOU_THRESHOLD = 0.7          
IMG_SIZE = 960            

# 2. 推論設備 (Device) 設定
INFERENCE_DEVICE = 'cuda:0' 

# 3. 類別對應 (單分類任務，Kaggle 要求 Class ID = 0)
CLASS_ID_MAPPING = {0: 0} 

# 4. 手動分批大小 (由於推論尺寸提高到 800，BATCH_SIZE 必須謹慎設定)
BATCH_SIZE = 32

# 5. *** 核心新增 ***: Test-Time Augmentation (TTA) 開關
USE_TTA = True # 啟用 TTA (強烈推薦)

# --- 輔助函數 (保持不變) ---

def denormalize_to_kaggle_format(x_center_norm, y_center_norm, w_norm, h_norm, img_w, img_h):
    """
    將 YOLO 的正規化座標 (x_c, y_c, w, h) 轉換為 Kaggle 要求的像素格式：
    (bb_left, bb_top, bb_width, bb_height)
    """
    
    x_center = x_center_norm * img_w
    y_center = y_center_norm * img_h
    width = w_norm * img_w
    height = h_norm * img_h

    bb_left = x_center - (width / 2)
    bb_top = y_center - (height / 2)

    bb_left = int(max(0, bb_left))
    bb_top = int(max(0, bb_top))
    # 保持寬高為 float 轉 int，確保與原始定義一致
    bb_width = int(width) 
    bb_height = int(height)
    
    return bb_left, bb_top, bb_width, bb_height

def generate_prediction_string(results, img_path):
    """
    處理 YOLOv11 推論結果，生成 Kaggle 要求的 PredictionString 格式。
    格式: <conf> <bb_left> <bb_top> <bb_width> <bb_height> <class> ...
    """
    
    prediction_list = []
    
    # 獲取圖片原始尺寸，用於反正規化座標
    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"Error reading image size for {os.path.basename(img_path)}: {e}")
        return ""

    # r.boxes 包含了邊界框、分數、類別等資訊
    for box in results.boxes:
        
        conf = box.conf.item()
        if conf < CONFIDENCE_THRESHOLD:
            continue
            
        # 注意: xywhn 是正規化後的 (x_center, y_center, w, h)
        xywhn = box.xywhn.tolist()[0]
        x_center_norm, y_center_norm, w_norm, h_norm = xywhn
        
        cls_id_yolo = int(box.cls.item())
        cls_id_kaggle = CLASS_ID_MAPPING.get(cls_id_yolo, -1) 
        
        if cls_id_kaggle == -1:
            continue
            
        bb_left, bb_top, bb_width, bb_height = denormalize_to_kaggle_format(
            x_center_norm, y_center_norm, w_norm, h_norm, img_w, img_h
        )
        
        # 構建單個物件的結果字串 (格式: conf bb_left bb_top bb_width bb_height class)
        obj_string = f"{conf:.4f} {bb_left} {bb_top} {bb_width} {bb_height} {cls_id_kaggle}"
        prediction_list.append(obj_string)
        
    return " ".join(prediction_list)

def extract_pure_id(filename):
    """從檔名 (e.g., '000001.jpg') 中提取純數字 ID (e.g., '1')，並移除前導零。"""
    base_name = os.path.splitext(filename)[0]
    try:
        # 使用 int() 轉換來移除前導零
        return str(int(base_name))
    except ValueError:
        return base_name 

def run_inference_and_export(best_model_path, test_image_dir, output_csv_file, inference_device):
    if not os.path.exists(best_model_path):
        print(f"錯誤: 模型權重未找到於 {best_model_path}")
        return

    print(f"從 {best_model_path} 載入模型...")
    model = YOLO(best_model_path).to(inference_device)
    
    if not os.path.isdir(test_image_dir):
        print(f"錯誤: 測試圖片目錄未找到於 {test_image_dir}")
        return

    image_paths = [os.path.join(test_image_dir, f) 
                   for f in os.listdir(test_image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_paths:
        print("錯誤: 在測試目錄中未找到任何圖片。")
        return

    # 按照 Image ID 數字順序排序
    def sort_key(path):
        try:
            return int(os.path.splitext(os.path.basename(path))[0])
        except ValueError:
            return os.path.basename(path)

    image_paths.sort(key=sort_key) 
    
    print(f"找到 {len(image_paths)} 張圖片。開始推論 (設備: {inference_device}, 尺寸: {IMG_SIZE}, TTA: {USE_TTA})...")

    submission_data = []
    total_images = len(image_paths)
    total_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE

    # *** 手動分批迭代的核心邏輯 ***
    for i in range(0, total_images, BATCH_SIZE):
        
        batch_paths = image_paths[i:i + BATCH_SIZE]
        current_batch_num = i // BATCH_SIZE + 1
        
        print(f"\n-> 正在推論批次 {current_batch_num}/{total_batches} ({len(batch_paths)} 張圖片)...")

        # 進行推論
        results = model.predict(
            source=batch_paths,           
            conf=CONFIDENCE_THRESHOLD, 
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE, 
            stream=False,                 
            verbose=False, 
            device=inference_device, 
            augment=USE_TTA # 應用 TTA
        )
        
        # 處理結果並收集數據
        for result in results:
            img_filename = os.path.basename(result.path)
            image_id_for_kaggle = extract_pure_id(img_filename)
            prediction_string = generate_prediction_string(result, result.path)
            
            submission_data.append({
                'Image_ID': image_id_for_kaggle, 
                'PredictionString': prediction_string
            })
        
        # 記憶體釋放
        if 'cuda' in inference_device:
            torch.cuda.empty_cache()
        gc.collect() 

    # 創建 Pandas DataFrame 並輸出為 CSV
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_csv_file, index=False, sep=',')
    
    print("\n--- 推論和導出任務完成 ---")
    print(f"成功導出 {len(submission_data)} 筆結果到 {output_csv_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and generate Kaggle submission file.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the best model weights file (e.g., runs/yolo11/final/weights/best.pt).")
    parser.add_argument('--test-image-dir', type=str, required=True, help="Path to the test image directory (e.g., data/test/img).")
    parser.add_argument('--output-csv', type=str, default='submission_final.csv', help="Name of the output CSV file for Kaggle submission.")
    parser.add_argument('--device', type=str, default='0', help="GPU device ID (e.g., '0' or '0,1') or 'cpu'.")
    
    args = parser.parse_args()
    run_inference_and_export(args.model_path, args.test_image_dir, args.output_csv, args.device)
