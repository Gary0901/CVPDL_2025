import os
import argparse
from PIL import Image

# --- 設定區塊 (Config Block) ---
CLASS_ID = 0  # 您的單一類別 ID，固定為 0
# --- 設定區塊 ---

def convert_bbox_to_yolo(x_min, y_min, width, height, img_w, img_h):
    """將像素坐標轉換為 YOLO 正規化格式 (center_x, center_y, w, h)"""
    x_center = (x_min + width / 2) / img_w
    y_center = (y_min + height / 2) / img_h
    w_norm = width / img_w
    h_norm = height / img_h
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

def main(gt_file, image_dir, yolo_labels_dir):
    # 創建標籤目錄
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # 1. 讀取 gt.txt 並分組標註
    annotations = {}
    print(f"Reading {gt_file}...")
    try:
        with open(gt_file, 'r') as f:
            for line in f:
                # 假設 gt.txt 中的數據是逗號分隔
                try:
                    # <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>
                    data = [float(x.strip()) for x in line.strip().split(',')]
                    if len(data) != 5:
                        print(f"Skipping malformed line: {line.strip()}")
                        continue
                        
                    frame, bb_left, bb_top, bb_width, bb_height = data
                    # 假設您的圖片命名格式是 8 位數字 (e.g., 00000001.jpg)
                    frame_name = f"{int(frame):08d}.jpg" 

                    # 將原始像素座標存儲起來，稍後再轉換
                    if frame_name not in annotations:
                        annotations[frame_name] = []
                    annotations[frame_name].append((bb_left, bb_top, bb_width, bb_height))
                    
                except Exception as e:
                    print(f"Skipping line due to error: {e}. Line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: GT file not found at {gt_file}")
        return

    # 2. 遍歷每個圖片，讀取尺寸並寫入 YOLO 標籤檔案
    print("Processing images and converting annotations...")
    for frame_name, bboxes in annotations.items():
        img_path = os.path.join(image_dir, frame_name)
        label_filename = frame_name.replace('.jpg', '.txt')
        label_path = os.path.join(yolo_labels_dir, label_filename)

        if not os.path.exists(img_path):
            # print(f"Warning: Image not found at {img_path}. Skipping.")
            continue

        try:
            # **自動讀取圖片寬度和高度**
            with Image.open(img_path) as img:
                IMAGE_W, IMAGE_H = img.size
            
            yolo_lines = []
            for bb_left, bb_top, bb_width, bb_height in bboxes:
                # 轉換並添加
                yolo_line = convert_bbox_to_yolo(bb_left, bb_top, bb_width, bb_height, IMAGE_W, IMAGE_H)
                yolo_lines.append(yolo_line)

            # 寫入 YOLO 標籤檔案
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
                
        except Exception as e:
            print(f"Error processing {frame_name}: {e}. Skipping conversion for this image.")
            
    print("\nData conversion complete. Labels are stored in the output directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert raw bounding box annotations to YOLO format.")
    parser.add_argument('--gt-file', type=str, required=True, help="Path to the original gt.txt file.")
    parser.add_argument('--image-dir', type=str, required=True, help="Path to the training image directory (e.g., data/train/img).")
    parser.add_argument('--output-labels-dir', type=str, default='yolo_labels', help="Directory to save the converted YOLO .txt label files.")
    
    args = parser.parse_args()
    main(args.gt_file, args.image_dir, args.output_labels_dir)
