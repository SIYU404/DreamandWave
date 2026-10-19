import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_classif

def main():
    print("=" * 65)
    print("【步驟六：腦波特徵貢獻度與正負向判斷依據分析】")
    print("=" * 65)

    csv_path = "eeg_features_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"錯誤：找不到 {csv_path}，請確認特徵檔存在。")
        return

    df = pd.read_csv(csv_path)
    df_binary = df[df['Emotion_Class'].isin(['Positive', 'Negative'])].copy()

    metadata_cols = ['Filename', 'PA_Score', 'NA_Score', 'Net_Valence', 'Emotion_Class']
    feature_cols = [c for c in df_binary.columns if c not in metadata_cols]

    # 對 PSD 特徵取對數 (Log Transform)，轉為 dB 尺度以解決 1e-13 數值下溢問題
    X_raw = df_binary[feature_cols].values
    X = np.log10(np.clip(X_raw, 1e-20, None))
    y = df_binary['Emotion_Class'].map({'Negative': 0, 'Positive': 1}).values

    # 1. 使用 ANOVA F-score 衡量單變量特徵區分能力
    f_scores, p_values = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    # 2. 同步訓練隨機森林並使用排列重要性 (Permutation Importance)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    perm_result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    importances = perm_result.importances_mean
    
    # 若排列重要性差異過小，則融合 F-score 作為特徵權重
    if importances.sum() == 0:
        importances = f_scores / (f_scores.sum() + 1e-9)
    else:
        importances = np.maximum(importances, 0)
        importances = importances / (importances.sum() + 1e-9)

    # 取得前 15 大關鍵特徵
    sorted_idx = np.argsort(importances)[::-1][:15]
    top_features = [feature_cols[i] for i in sorted_idx]
    top_scores = importances[sorted_idx]

    print("\n--- 影響模型判斷正負向的前 15 大關鍵腦波特徵排行榜 ---")
    print(f"{'排名':<5}{'特徵名稱 (電極導聯_頻段)':<28}{'貢獻權重 (%)':<15}{'正向 vs 負向趨勢'}")
    print("-" * 65)

    pos_means = df_binary[df_binary['Emotion_Class'] == 'Positive'][top_features].mean()
    neg_means = df_binary[df_binary['Emotion_Class'] == 'Negative'][top_features].mean()

    for rank, (feat, score) in enumerate(zip(top_features, top_scores), 1):
        p_val = pos_means[feat]
        n_val = neg_means[feat]
        if p_val > n_val:
            trend = f"正向較高 (正:{p_val:.2e} > 負:{n_val:.2e})"
        else:
            trend = f"負向較高 (負:{n_val:.2e} > 正:{p_val:.2e})"
        print(f"{rank:<5}{feat:<28}{score*100:>6.2f}%        {trend}")

    # 3. 彙總五大頻段的總體決策影響力
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    band_importance = {b: 0.0 for b in band_names}
    for feat, imp in zip(feature_cols, importances):
        for b in band_names:
            if b in feat:
                band_importance[b] += imp
                break

    total_band_imp = sum(band_importance.values())
    if total_band_imp > 0:
        band_importance = {b: v / total_band_imp for b, v in band_importance.items()}

    print("\n--- 五大腦波頻段總體決策影響力分佈 ---")
    for b, imp in sorted(band_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {b:<8} 頻段: {imp*100:.2f}%")

    # 4. 繪製圖表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 圖 A: 前 10 大關鍵特徵長條圖
    sns.barplot(
        ax=axes[0], 
        x=top_scores[:10], 
        y=top_features[:10], 
        hue=top_features[:10], 
        palette='Blues_r', 
        legend=False
    )
    axes[0].set_title("Top 10 Decisive EEG Features", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Relative Importance Weight")

    # 圖 B: 各頻段佔比長條分佈圖（避免圓餅圖 0 尺寸錯誤）
    bands = list(band_importance.keys())
    band_vals = [band_importance[b] for b in bands]
    sns.barplot(
        ax=axes[1], 
        x=bands, 
        y=band_vals, 
        hue=bands, 
        palette='crest', 
        legend=False
    )
    axes[1].set_title("Overall Importance by Frequency Band", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Normalized Weight")
    axes[1].set_ylim(0, max(band_vals) * 1.2 if max(band_vals) > 0 else 1.0)

    plt.tight_layout()
    chart_filename = "feature_explanation_analysis.png"
    plt.savefig(chart_filename, dpi=300)
    print("\n" + "=" * 65)
    print(f"特徵解析圖表已儲存至: {os.path.abspath(chart_filename)}")
    print("=" * 65)
    plt.show(block=True)

if __name__ == "__main__":
    main()