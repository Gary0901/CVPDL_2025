import os
import random
import shutil

# --- 設定 ---
# 1. 原始資料夾路徑
SOURCE_IMAGE_DIR = "../data/CVPDL_hw2/CVPDL_hw2/train"
SOURCE_LABEL_DIR = "../data/train_yolo_labels"

# 2. 新的資料集根目錄
DEST_DATASET_DIR = "../data/datasets"


# 3. 驗證集分割比例 (例如 0.2 代表 20% 的資料將作為驗證集)
VAL_SPLIT_RATIO = 0.2

# 支援的圖片副檔名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# --- 結束設定 ---


def create_dirs():
    """
    建立 YOLOv8 所需的目標資料夾結構
    """
    global TRAIN_IMAGE_PATH, VAL_IMAGE_PATH, TRAIN_LABEL_PATH, VAL_LABEL_PATH
    
    TRAIN_IMAGE_PATH = os.path.join(DEST_DATASET_DIR, "images", "train")
    VAL_IMAGE_PATH = os.path.join(DEST_DATASET_DIR, "images", "val")
    TRAIN_LABEL_PATH = os.path.join(DEST_DATASET_DIR, "labels", "train")
    VAL_LABEL_PATH = os.path.join(DEST_DATASET_DIR, "labels", "val")
    
    os.makedirs(TRAIN_IMAGE_PATH, exist_ok=True)
    os.makedirs(VAL_IMAGE_PATH, exist_ok=True)
    os.makedirs(TRAIN_LABEL_PATH, exist_ok=True)
    os.makedirs(VAL_LABEL_PATH, exist_ok=True)
    
    print("資料夾結構建立完成：")
    print(f"- {TRAIN_IMAGE_PATH}")
    print(f"- {VAL_IMAGE_PATH}")
    print(f"- {TRAIN_LABEL_PATH}")
    print(f"- {VAL_LABEL_PATH}")

def split_data():
    """
    執行資料分割與檔案複製
    """
    print(f"\n正在從 {SOURCE_IMAGE_DIR} 讀取圖片...")
    
    # 讀取所有圖片檔案，並過濾掉非圖片檔
    image_files = []
    for f in os.listdir(SOURCE_IMAGE_DIR):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            image_files.append(f)
            
    if not image_files:
        print(f"錯誤：在 {SOURCE_IMAGE_DIR} 中找不到任何圖片檔案。")
        return
        
    # 隨機打亂
    random.shuffle(image_files)
    
    # 計算分割點
    total_files = len(image_files)
    val_count = int(total_files * VAL_SPLIT_RATIO)
    train_count = total_files - val_count
    
    # 分割檔案列表
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    
    print(f"總檔案數: {total_files}")
    print(f"訓練集大小: {len(train_files)} ({(1-VAL_SPLIT_RATIO)*100:.0f}%)")
    print(f"驗證集大小: {len(val_files)} ({VAL_SPLIT_RATIO*100:.0f}%)")

    # 定義一個輔助函式來複製檔案
    def copy_file_pairs(file_list, dest_image_dir, dest_label_dir):
        copied_count = 0
        for image_filename in file_list:
            base_name = os.path.splitext(image_filename)[0]
            label_filename = base_name + ".txt"
            
            # 原始路徑
            src_image = os.path.join(SOURCE_IMAGE_DIR, image_filename)
            src_label = os.path.join(SOURCE_LABEL_DIR, label_filename)
            
            # 目標路徑
            dest_image = os.path.join(dest_image_dir, image_filename)
            dest_label = os.path.join(dest_label_dir, label_filename)
            
            # 檢查圖片和標籤是否都存在
            if os.path.exists(src_image) and os.path.exists(src_label):
                try:
                    # !! 使用 shutil.copy() 而不是 shutil.move() !!
                    shutil.copy(src_image, dest_image)
                    shutil.copy(src_label, dest_label)
                    copied_count += 1
                except Exception as e:
                    print(f"錯誤：複製 {image_filename} 時發生問題: {e}")
            else:
                if not os.path.exists(src_label):
                    print(f"警告：找不到對應的標籤檔 {label_filename}，跳過 {image_filename}")
                if not os.path.exists(src_image):
                     print(f"警告：找不到圖片檔 {src_image} (這不應該發生)")
        return copied_count

    # 執行複製
    print("\n正在複製訓練集檔案...")
    train_copied = copy_file_pairs(train_files, TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH)
    print(f"成功複製 {train_copied} 對訓練檔案。")

    print("\n正在複製驗證集檔案...")
    val_copied = copy_file_pairs(val_files, VAL_IMAGE_PATH, VAL_LABEL_PATH)
    print(f"成功複製 {val_copied} 對驗證檔案。")
    
    print("\n資料分割完成！")
    print(f"原始資料夾 {SOURCE_IMAGE_DIR} 和 {SOURCE_LABEL_DIR} 中的檔案已完整保留。")


if __name__ == "__main__":
    create_dirs()
    split_data()