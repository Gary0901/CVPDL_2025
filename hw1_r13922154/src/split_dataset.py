import os
import shutil
import random
import argparse
from sklearn.model_selection import train_test_split

def split_and_move_files(source_image, source_label, target_root, split_ratio, seed):
    """將圖片和對應標籤檔隨機分割並移動到 train/val 目錄"""

    if not os.path.isdir(source_image):
        print(f"Error: Source image directory not found at {source_image}")
        return
    if not os.path.isdir(source_label):
        print(f"Error: Source label directory not found at {source_label}")
        return

    # 獲取所有圖片檔案列表 (假設都是 .jpg 且與標籤檔名一致)
    all_image_files = [f for f in os.listdir(source_image) if f.lower().endswith('.jpg')]

    if not all_image_files:
        print(f"Error: No .jpg files found in {source_image}. Check your path.")
        return
        
    print(f"Total files found: {len(all_image_files)}")
    
    # 執行隨機分割，test_size=split_ratio 即 val_size
    train_files, val_files = train_test_split(
        all_image_files, 
        test_size=split_ratio, 
        random_state=seed
    )
    
    print(f"Training set size: {len(train_files)}")
    print(f"Validation set size: {len(val_files)}")

    # 設置目標目錄
    target_dirs = {
        'train': {
            'image': os.path.join(target_root, 'images', 'train'),
            'label': os.path.join(target_root, 'labels', 'train')
        },
        'val': {
            'image': os.path.join(target_root, 'images', 'val'),
            'label': os.path.join(target_root, 'labels', 'val')
        }
    }

    # 創建所有目標目錄
    for subset in ['train', 'val']:
        os.makedirs(target_dirs[subset]['image'], exist_ok=True)
        os.makedirs(target_dirs[subset]['label'], exist_ok=True)

    def move_files(file_list, subset):
        count = 0
        for image_name in file_list:
            # 1. 處理圖片檔
            src_img_path = os.path.join(source_image, image_name)
            dst_img_path = os.path.join(target_dirs[subset]['image'], image_name)
            
            # 2. 處理標籤檔
            label_name = image_name.replace('.jpg', '.txt') # 假設圖片是 .jpg
            src_lbl_path = os.path.join(source_label, label_name)
            dst_lbl_path = os.path.join(target_dirs[subset]['label'], label_name)
            
            # 檢查標籤檔是否存在
            if not os.path.exists(src_lbl_path):
                print(f"Warning: Missing label for {image_name}. Skipping both files.")
                continue

            # 複製檔案（使用 copy 確保原始檔案保留）
            try:
                shutil.copy(src_img_path, dst_img_path)
                shutil.copy(src_lbl_path, dst_lbl_path)
                count += 1
            except FileNotFoundError as e:
                print(f"Error: {e}. Skipping file.")
        
        print(f"Successfully moved/copied {count} file pairs to {subset}.")

    print("\nMoving/Copying Training files...")
    move_files(train_files, 'train')
    
    print("\nMoving/Copying Validation files...")
    move_files(val_files, 'val')

    print("\nDataset split and organization complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split images and YOLO labels into train/val sets.")
    parser.add_argument('--source-image-dir', type=str, required=True, help="Root directory of source images.")
    parser.add_argument('--source-label-dir', type=str, required=True, help="Root directory of converted YOLO labels.")
    parser.add_argument('--target-root-dir', type=str, default='yolo_dataset', help="Root directory for the output train/val structure.")
    parser.add_argument('--split-ratio', type=float, default=0.20, help="Validation set proportion (e.g., 0.20 for 20%%).")
    parser.add_argument('--random-seed', type=int, default=42, help="Random seed for reproducible splitting.")
    
    args = parser.parse_args()
    
    split_and_move_files(
        args.source_image_dir, 
        args.source_label_dir, 
        args.target_root_dir, 
        args.split_ratio, 
        args.random_seed
    )
