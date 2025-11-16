# CVPDL 課程作業列表 (Course Assignments List)

本倉庫收錄 CVPDL 課程（Computer Vision Practice with Deep Learning）的所有作業，專注於深度學習在電腦視覺中的實踐。

## 作業一覽 (Assignments Overview)

| 作業編號 | 主題 (Topic) | 權重 |
| :---: | :--- | :---: |
| **HW1** | Object Detection (物件偵測) | 15% |
| **HW2** | Long-tailed Object Detection (長尾物件偵測) | 15% |
| **HW3** | Image Generation (影像生成) | 20% |

---

## 作業說明與挑戰 (Assignment Description and Challenges)

### **[HW1] Object Detection**

* **作業主題:** 針對「Group-housed Swine (群居豬隻)」的物件偵測 。
* **目標:** 在提供的影像中準確識別並定位豬隻。
* **數據集:** 包含單一類別（pig）的訓練集和測試集。
* **核心挑戰:** 在擁擠、視覺遮擋較高的環境（如群居）中，實現高精確度的物件定位。
* **評估指標:** $mAP_{50:95}$。
* **報告重點:** 說明模型架構、實作細節，並分析量化改進和視覺化結果。

### **[HW2] Long-tailed Object Detection**

* **作業主題:** 針對「無人機影像中物件」的長尾物件偵測。
* **目標:** 實作物件偵測模型，用於識別四個類別：car, hov, person, motorcycle。
* **數據集:** 影像來自無人機視角，數據集類別分佈呈現極度不平衡的「長尾」現象。
* **核心挑戰 (難點):**
    1.  **長尾分佈 (Long-Tailed):** 這是主要難點。需要設計或採用特殊策略來處理少數類別（尾部）數據不足導致的偵測性能下降。
    2.  **無人機視角:** 可能涉及俯視角、小物件偵測和高密度場景（如停車場）。
* **評估指標:** 平均精確度 $mAP_{50:95}$。
* **報告重點:** **必須分析您的方法如何解決長尾問題**，並提供改進前後的比較結果，以及模型描述和實作細節 。

### **[HW3] Image Generation**

* **作業主題:** 影像生成(Image Generation)。
* **目標:** 實作Diffusion-based Generative Model 於手寫數字的生成。
* **數據集:** MNIST (60000 張 28 X 28 RGB圖像）。
* **核心挑戰 (難點):**
    1. 實作 Diffusion Generative Models。
    2. 訓練模型，在生成10000張影像後，達成優異的FID評估分數。
* **評估指標:** Frechet Inception Distance (FID)。
* **報告重點:** 模型描述、實作細節、量化分析，以及 Diffusion Process Visualizations (視覺化擴散過程) 。