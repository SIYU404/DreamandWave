import os
import glob
import pandas as pd

def find_file(filename):
    """優先檢查 Data/ 目錄，若無則搜尋當前目錄與所有子目錄"""
    direct_path = os.path.join(".", "Data", filename)
    if os.path.exists(direct_path):
        return direct_path
    
    matches = glob.glob(os.path.join(".", "**", filename), recursive=True)
    if matches:
        return matches[0]
    return filename

def inspect_emotion_ratings():
    ratings_path = find_file("Ratings.csv")

    if not os.path.exists(ratings_path):
        print(f"錯誤：找不到標籤檔案 {ratings_path}，請確認是否位於 Data/ 資料夾內。")
        return

    print("=" * 60)
    print("【步驟二：夢境情緒問卷與標籤解析】")
    print(f"讀取路徑: {ratings_path}")
    print("=" * 60)

    df_ratings = pd.read_csv(ratings_path)
    print(f"成功載入 Ratings.csv，共 {len(df_ratings)} 筆夢境評分紀錄。")

    # 定義受試者自評的正向 (PA) 與負向 (NA) 欄位名稱 (各 10 項，每項 0 ~ 4 分)
    pa_cols = [f"SR_PA{i}" for i in range(1, 11)]
    na_cols = [f"SR_NA{i}" for i in range(1, 11)]

    # 1. 計算正向總分 (PA_Score)、負向總分 (NA_Score) 與淨情緒效價 (Net_Valence)
    df_ratings['PA_Score'] = df_ratings[pa_cols].sum(axis=1)
    df_ratings['NA_Score'] = df_ratings[na_cols].sum(axis=1)
    df_ratings['Net_Valence'] = df_ratings['PA_Score'] - df_ratings['NA_Score']

    # 2. 定義情緒三分類標籤
    def categorize_emotion(row):
        if row['PA_Score'] > row['NA_Score']:
            return "Positive (正向)"
        elif row['NA_Score'] > row['PA_Score']:
            return "Negative (負向)"
        else:
            return "Neutral (中性/平衡)"

    df_ratings['Emotion_Label'] = df_ratings.apply(categorize_emotion, axis=1)

    # 3. 顯示情緒分佈統計
    print("\n--- 夢境情緒類別統計分佈 ---")
    print(df_ratings['Emotion_Label'].value_counts().to_string())

    print("\n--- 正向與負向平均得分 ---")
    print(f"正向 (PA) 平均得分: {df_ratings['PA_Score'].mean():.2f} / 40 分")
    print(f"負向 (NA) 平均得分: {df_ratings['NA_Score'].mean():.2f} / 40 分")

    # 4. 展示前 5 筆樣本對應資訊
    print("\n--- 前 5 筆檔案情緒標籤範例 ---")
    preview_cols = ['Filename', 'DreamReport_Wordcount', 'PA_Score', 'NA_Score', 'Emotion_Label']
    print(df_ratings[preview_cols].head(5).to_string(index=False))
    print("=" * 60)

    # 5. 互動查詢功能：輸入 EDF 檔名查詢情緒分數
    while True:
        target = input("\n請輸入想查詢的腦波檔名 (例如 case02_s10.edf 或 case100_s23.edf，輸入 q 離開): ").strip()
        if target.lower() == 'q':
            print("結束查詢。")
            break

        clean_target = os.path.basename(target)
        if not clean_target.endswith(".edf"):
            clean_target += ".edf"

        matched = df_ratings[df_ratings['Filename'] == clean_target]
        if not matched.empty:
            row = matched.iloc[0]
            print(f"\n[查詢結果] 檔案: {row['Filename']}")
            print(f"  夢境報告字數: {row['DreamReport_Wordcount']} 字")
            print(f"  正向總分 (PA): {row['PA_Score']} | 負向總分 (NA): {row['NA_Score']}")
            print(f"  判定情緒類別: {row['Emotion_Label']}")
            print(f"  高評分正向項目 (>=2分): {[col for col in pa_cols if row[col] >= 2]}")
            print(f"  高評分負向項目 (>=2分): {[col for col in na_cols if row[col] >= 2]}")
        else:
            print(f"在 Ratings.csv 中找不到檔名為 '{clean_target}' 的紀錄。")

if __name__ == "__main__":
    inspect_emotion_ratings()