# Animal-vs-Urban-Sound-CNN
Sound Classification with Mel Spectrogram &amp; CNN
本專案使用 [ESC-50]開源資料集中的動物聲音與城市噪音，進行二元音訊分類任務。
透過 Mel-spectrogram 特徵擷取與`CNN`，在有限資料條件下成功建立一個準確且高效的分類模型。

為展示用途，`sample/` 目錄下包含幾筆來自 ESC-50 的聲音樣本，涵蓋兩類別：

- `Animals` 類別（如貓叫、狗吠）
- `Urban` 類別（如汽車聲、喇叭聲）

這些檔案僅供範例使用，完整資料請參考 [ESC-50](https://github.com/karoldvl/ESC-50)。
---
## 專案動機

在本專案之前，曾使用傳統機器學習模型 **Random Forest** 進行分類任務，達成以下測試結果：

- Accuracy: 86.72%
- Precision: 87.25%
- F1-Score: 86.53%

雖然結果尚可接受，但希望透過 **改用深度學習模型 CNN** 來提升效能，因此著手設計本系統。

本研究同時也作為探索深度學習於音訊分類實務應用的學習與實驗。

---

## 資料預號與準備

- 資料來源：ESC-50 音訊集 (Animals及Urban)
- 採樣率：44100 Hz
- 音訊長度：5 秒

### 特徵提取

- 設定：
  - n_mels = 32
  - n_fft = 256
  - hop_length = 64
- 此外 **Mel-spectrograms** 會使用 z-score 正規化 (mean/std)

### 資料增強

只在訓練資料上進行：

- 随機增加波動壓力聲
- 随機時間位移
- 随機音量調整

### 資料分割

- 80% 訓練
- 10% 驗證
- 10% 測試

---

## CNN 模型組成

使用 TensorFlow / Keras 簡易 API 實作：

- 2 個 Convolutional Block (包含 Conv2D + BatchNorm + MaxPooling + Dropout)
- GlobalAveragePooling
- Dense Layer + Dropout
- 輸出 layer 為 softmax 

訓練使用:

- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam (lr=0.0001)`

### 分類不均平處理

- 使用 `sklearn.utils.compute_class_weight` 來正規各類範例比例

### 訓練設定

- Epochs: 30
- Batch size: 10 (每類 5 個样本)
- Callbacks:
  - EarlyStopping (patience=10, monitor val_loss)
  - ModelCheckpoint (save best model by val_loss, mode='min')
  - ReduceLROnPlateau (patience=7)

---

## 訓練效果

### 訓練曲線
![訓練曲線圖](/results/training_history.png)

- 訓練在第 20 回合左右已收斂較佳
- Epoch 21 後 validation loss 波動上升，但透過 callbacks設定 儲存最佳模型

---

## 測試結果

### 混淆矩陣
![混淆矩陣](/results/confusion_matrix.png)

### 分類效能報告:

```
              precision    recall  f1-score   support

     Animals       0.94      0.92      0.93        37
urban noises       0.89      0.93      0.91        27

    accuracy                           0.92        64
   macro avg       0.92      0.92      0.92        64
weighted avg       0.92      0.92      0.92        64
```

### 分析要點:

- 模型達成 **92% 全體正確率**，相較先前 Random Forest 提升明顯
- Urban noises precision 較前一版本略有改善，類別平衡更佳
- CNN 模型顯示出穩定的泛化能力與分類表現

---

## 未來擴展方向

- 對 CNN 組成進行設定調優 (ex. filter 數量, dropout 比例)
- 增加更多模擬噪音或增強策略
- 嘗試更複雜的深度模型 (CRNN, ResNet 等)
- 擴展至 ESC-50 全類別或 multi-class 分類

---

## 系統環境

- Python 3.9.13
- TensorFlow 2.13.0
- librosa 0.11.0 
- matplotlib 3.9.4 
- seaborn 0.13.2 
- scikit-learn 1.6.1 

---

## 參考資料

- [ESC-50 Dataset](https://github.com/karoldvl/ESC-50)
- [Librosa Documentation](https://librosa.org/)
- [TensorFlow](https://www.tensorflow.org/)
