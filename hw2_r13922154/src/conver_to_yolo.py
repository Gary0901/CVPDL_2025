import os
import cv2

# --- 設定你的路徑 ---

# 1. 原始資料夾：包含圖片 (img0001.png) 和原始標籤 (img0001.txt)
INPUT_DIR = "../data/CVPDL_hw2/CVPDL_hw2/train" 

# 2. 輸出資料夾：儲存轉換後的 YOLO 格式標籤
OUTPUT_DIR = "../data/train_yolo_labels"

# ----------------------

# 確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 支援的圖片格式
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# 遍歷 INPUT_DIR 中的所有檔案
print(f"開始轉換... 來源: '{INPUT_DIR}', 輸出: '{OUTPUT_DIR}'")
file_list = os.listdir(INPUT_DIR)

for filename in file_list:
    # 檢查是否為原始標籤 .txt 檔
    if filename.endswith(".txt"):
        base_filename = os.path.splitext(filename)[0] # e.g., "img0001"
        gt_txt_path = os.path.join(INPUT_DIR, filename)
        
        # 1. 尋找對應的圖片檔案
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            potential_image_path = os.path.join(INPUT_DIR, base_filename + ext)
            if os.path.exists(potential_image_path):
                image_path = potential_image_path
                break
        
        # 如果找不到圖片
        if image_path is None:
            print(f"警告：找到了 {filename}，但找不到對應的圖片檔案。跳過此檔案。")
            continue

        # 2. 讀取圖片尺寸
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"錯誤：無法讀取圖片 {image_path}。跳過 {filename}。")
                continue
            
            image_height, image_width, _ = image.shape
        except Exception as e:
            print(f"錯誤：讀取 {image_path} 尺寸時發生問題: {e}。跳過 {filename}。")
            continue

        # 準備儲存轉換後的 YOLO 標籤
        yolo_labels = []
        
        # 3. 讀取原始 gt.txt
        try:
            with open(gt_txt_path, 'r') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split(',')
                    if len(parts) != 5:
                        print(f"警告：{filename} 中的行格式錯誤: {line}。跳過此行。")
                        continue
                    
                    # 讀取來源格式
                    class_label = int(parts[0])
                    tl_x = float(parts[1])
                    tl_y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # 4. 執行轉換計算
                    x_center = tl_x + (w / 2)
                    y_center = tl_y + (h / 2)
                    
                    # 歸一化
                    x_center_norm = x_center / image_width
                    y_center_norm = y_center / image_height
                    w_norm = w / image_width
                    h_norm = h / image_height
                    
                    # 格式化為 YOLO 字串
                    yolo_line = f"{class_label} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                    yolo_labels.append(yolo_line)
                    
        except Exception as e:
            print(f"錯誤：處理 {gt_txt_path} 時發生問題: {e}。跳過此檔案。")
            continue

        # 5. 寫入新的 YOLO 格式 txt 檔案
        output_txt_path = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(output_txt_path, 'w') as f_out:
                for line in yolo_labels:
                    f_out.write(line + "\n")
        except Exception as e:
            print(f"錯誤：無法寫入 {output_txt_path}: {e}。")

print(f"轉換完成！YOLO 格式的標籤檔已儲存在 '{OUTPUT_DIR}'。")