import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

def main():
    print("=" * 60)
    print("【步驟五：訓練模型並打包存檔】")
    print("=" * 60)

    csv_path = "eeg_features_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"錯誤：找不到 {csv_path}，請確認步驟四已產出該檔案。")
        return

    df = pd.read_csv(csv_path)
    # 取出 Positive 與 Negative 樣本
    df_binary = df[df['Emotion_Class'].isin(['Positive', 'Negative'])].copy()

    metadata_cols = ['Filename', 'PA_Score', 'NA_Score', 'Net_Valence', 'Emotion_Class']
    feature_cols = [c for c in df_binary.columns if c not in metadata_cols]

    X = df_binary[feature_cols].values
    y = df_binary['Emotion_Class'].map({'Negative': 0, 'Positive': 1}).values

    print(f"訓練樣本數: {len(X)} 筆 (Positive: {sum(y==1)}, Negative: {sum(y==0)})")
    print(f"使用特徵數: {len(feature_cols)} 個頻段特徵")

    # 建立支援向量機分類管線
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42))
    ])

    # 5 折交叉驗證檢驗效能
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='balanced_accuracy')
    print(f"5-Fold 交叉驗證平衡準確率: {scores.mean()*100:.2f}% (± {scores.std()*100:.2f}%)")

    # 用全部資料完整擬合
    model_pipeline.fit(X, y)

    # 打包儲存模型與特徵名稱
    save_data = {
        'model': model_pipeline,
        'feature_names': feature_cols
    }
    model_file = "eeg_emotion_model.pkl"
    joblib.dump(save_data, model_file)

    print("\n" + "=" * 60)
    print(f"模型訓練完成並已匯出至: {os.path.abspath(model_file)}")
    print("=" * 60)

if __name__ == "__main__":
    main()