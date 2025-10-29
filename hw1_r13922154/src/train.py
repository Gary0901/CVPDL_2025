import os
import time
from datetime import timedelta
from ultralytics import YOLO
import torch # 導入 PyTorch 進行權重操作

# --- 配置區 ---
# 1. 數據與模型設定    
MODEL_CONFIG = 'yolo11x.yaml' 

# 3. 超參數 (Hyperparameters)
HYPERPARAMETERS = {
    'epochs': 200,            
    'batch': 16,              
    'imgsz': 960,             
    'optimizer': 'AdamW',     
    'lr0': 0.0005 ,           
    'lrf': 0.00001,           
    'momentum': 0.937,        
    'weight_decay': 0.0005,   
    'patience': 100,          
    
    # 損失函數權重調整：增加邊界框 (Box) 損失權重
    'box': 10.0,  # 原為 7.5，增加以提高定位精度
    
    # NMS 閾值調整：降低 NMS IoU 閾值
    'iou': 0.6,   # 原為 0.7，降低以保留密集重疊的偵測框
    
    # 資料增廣調整
    'hsv_h': 0.015,           
    'hsv_s': 0.7,             
    'hsv_v': 0.4,             
    'degrees': 10.0,           
    'translate': 0.1,         
    'scale': 0.5,             
    'shear': 0.0,             
    'fliplr': 0.5,            
    'copy_paste': 0.05,
    'cutmix' : 0.0,
    'mosaic' : 0.5,
    'close_mosaic': 30, # 延後關閉 Mosaic 增強，讓模型在訓練後期更早地在真實圖像上穩定學習
    
    'single_cls':True,
    'cos_lr':True,   
    
}

# --- 核心邏輯修改：使用排除法載入 Backbone 權重 ---
def load_backbone_weights_only(model: YOLO, pt_path: str):
    """
    使用排除 Neck/Head 模塊索引的策略，載入預訓練模型檔案中的 Backbone 權重。
    Args:
        model (YOLO): 從 .yaml 初始化的 YOLO 模型實例。
        pt_path (str): 預訓練權重檔案 (.pt) 的路徑。
    """
    if not os.path.exists(pt_path):
        print(f"警告：預訓練權重檔案未找到於 {pt_path}。將使用完全隨機初始化的權重進行訓練。")
        return

    print(f"\n--- 權重載入檢查開始 (使用排除法) ---")
    
    try:
        # 1. 載入完整的預訓練權重
        ckpt = torch.load(pt_path, map_location='cpu')
        pretrained_state_dict = ckpt['model'].float().state_dict() # 提取模型權重字典
    except Exception as e:
        print(f"錯誤：載入預訓練權重時發生錯誤: {e}")
        return

    # 2. 定義要排除的 Neck/Head 模塊索引 (10 到 23)
    LAST_INDICES = list(range(10, 24)) 
    excluded_prefixes = tuple(f"model.{i}." for i in LAST_INDICES)

    # 3. 過濾權重：只保留 Backbone 權重
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items() 
        if not k.startswith(excluded_prefixes)
    }

    print(f"原始預訓練權重數量: {len(pretrained_state_dict)}")
    print(f"排除 Neck/Head 後的 Backbone 權重數量: {len(filtered_state_dict)}")
    
    # 4. 將過濾後的 Backbone 權重載入到新模型中
    try:
        missing, unexpected = model.model.load_state_dict(filtered_state_dict, strict=False)
        
        print("\n--- 載入結果總結 ---")
        print(f"Missing (保持隨機的 Neck/Head 權重數量): {len(missing)}")
        print(f"Unexpected (在目標模型中不存在的預訓練權重數量): {len(unexpected)}")
        print("Backbone 權重載入完成。Neck/Head 保持隨機初始化。")
    
    except RuntimeError as e:
        print(f"\n載入權重時發生 RuntimeError，通常是 Backbone 結構不匹配: {e}")
    
    print("------------------------------------------\n")

def train_model(data_yaml, pretrained_pt, device, project_name, experiment_name):
    print(f"Starting YOLO training with partial scratch initialization. Model: {MODEL_CONFIG} (Structure)")
    print(f"Data config: {data_yaml}")

    try:
        # 1. 載入模型配置 (使用 .yaml 確保 Neck/Head 隨機初始化)
        model = YOLO(MODEL_CONFIG)
    except FileNotFoundError:
        print(f"Error: Model config file {MODEL_CONFIG} not found.")
        return
    
    # 2. 執行部分權重載入
    load_backbone_weights_only(model, pretrained_pt)
    
    TOTAL_EPOCHS = HYPERPARAMETERS.get('epochs', 0)
    start_time = time.time()
    
    print("--- 訓練開始 ---")
    
    # 3. 開始訓練
    model.train(
        data=data_yaml,
        project=project_name,
        name=experiment_name,
        device=device,
        
        # 傳遞超參數
        **HYPERPARAMETERS
    )
    
    # --- 訓練完成後，計算總時間 ---
    end_time = time.time()
    total_duration = end_time - start_time
    
    if TOTAL_EPOCHS > 0:
        avg_epoch_time = total_duration / TOTAL_EPOCHS
        print("\n--- 訓練結果總結 ---")
        print(f"總共訓練了 {TOTAL_EPOCHS} 個 Epochs")
        print(f"訓練總時長: {timedelta(seconds=total_duration)}")
        print(f"平均每個 Epoch 所需時間: {timedelta(seconds=avg_epoch_time)}")
    else:
        print("訓練未運行或 Epoch 數為零。")

    print("\n訓練完成！結果保存在:", os.path.join(project_name, experiment_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv11x with customized partial weight loading and hyperparameters.")
    parser.add_argument('--data-yaml', type=str, required=True, help="Path to the YOLO data configuration YAML file (e.g., pigs.yaml).")
    parser.add_argument('--pretrained-pt', type=str, default='yolo11x.pt', help="Path to the YOLOv11x pretrained weights file.")
    parser.add_argument('--device', type=str, default='0', help="GPU device ID (e.g., '0' or '0,1') or 'cpu'.")
    parser.add_argument('--project-name', type=str, default='runs/yolo11', help="Project directory name for saving results.")
    parser.add_argument('--experiment-name', type=str, default='yolo11x_final', help="Experiment name for the current run.")
    
    args = parser.parse_args()
    train_model(args.data_yaml, args.pretrained_pt, args.device, args.project_name, args.experiment_name)
