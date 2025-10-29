from ultralytics import YOLO

def main():
    # --- 1. 更改模型：從 YOLO('yolov8m.yaml') 改為 YOLOv10('yolov10m.yaml') ---
    print("正在從 yolov10x.yaml 載入模型架構 (從頭開始訓練)...")
    # 你可以根據需求選擇 yolov10n, yolov10s, yolov10m, yolov10b, yolov10l, yolov10x
    model = YOLO('yolo11x.yaml')

    # --- 自訂你的輸出路徑 ---
    OUTPUT_PROJECT_FOLDER = "my_yolo_experiments"
    EXPERIMENT_NAME = "experiments1" # 建議更改實驗名稱以作區分
    # -------------------------
    
    # --- Early Stopping 設定 ---
    PATIENCE_EPOCHS = 30 
    
    print(f"開始訓練... 實驗將儲存在：{OUTPUT_PROJECT_FOLDER}/{EXPERIMENT_NAME}")
    
    results = model.train(
        data='hw2_dataset.yaml',  
        epochs=300,                
        imgsz=1440,                 
        batch=4,
        device=0,  # 使用 GPU，請確認你的 GPU ID
        lr0=0.006,
        
        # --- 2. 啟用 Focal Loss 的新方法 ---
        # 移除舊的 FocalLossTrainer，直接設定 fl_gamma 參數
        # fl_gamma > 0.0 即會啟用 Focal Loss。常見值為 1.5 或 2.0
        dfl=1.5,
        # ------------------------------------
        
        # data augmentation
        
        # hsv_h=0.015,
        # hsv_s=0.7,
        # hsv_v=0.4,
        # mosaic=1.0,
        mixup=0.1,
        
        cls=1.5, # 分類損失的權重
        weight_decay=0.001,
        cos_lr=True,
        
        # --- Early Stopping 設定 ---
        patience=PATIENCE_EPOCHS,
        
        # --- 指定輸出資料夾和名稱 ---
        project=OUTPUT_PROJECT_FOLDER,
        name=EXPERIMENT_NAME,
        # -------------------------------
        
        # 從頭開始訓練，不載入預訓練權重
        pretrained=False, 
    )
    print("訓練完成！")
    
    # results.save_dir 會自動回傳完整的路徑
    print(f"模型和結果儲存在：{results.save_dir}")

if __name__ == '__main__':
    main()
