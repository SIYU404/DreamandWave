import os
import glob
import mne
import pandas as pd
import matplotlib.pyplot as plt

def find_file(filename):
    """優先檢查 Data/，否則搜尋當前目錄與所有子目錄"""
    direct_path = os.path.join(".", "Data", filename)
    if os.path.exists(direct_path):
        return direct_path
    matches = glob.glob(os.path.join(".", "**", filename), recursive=True)
    return matches[0] if matches else filename

def get_edf_files():
    """遞迴搜尋所有 .edf 檔案"""
    files = glob.glob(os.path.join(".", "**", "*.edf"), recursive=True)
    return sorted(files)

def load_ratings():
    """讀取 Ratings.csv 並預先計算好情緒指標"""
    ratings_path = find_file("Ratings.csv")
    if not os.path.exists(ratings_path):
        print(f"警告：找不到標籤檔 {ratings_path}")
        return None

    df = pd.read_csv(ratings_path)
    pa_cols = [f"SR_PA{i}" for i in range(1, 11)]
    na_cols = [f"SR_NA{i}" for i in range(1, 11)]

    df['PA_Score'] = df[pa_cols].sum(axis=1)
    df['NA_Score'] = df[na_cols].sum(axis=1)

    def categorize_emotion(row):
        if row['PA_Score'] > row['NA_Score']:
            return "Positive (正向)"
        elif row['NA_Score'] > row['PA_Score']:
            return "Negative (負向)"
        else:
            return "Neutral (中性/平衡)"

    df['Emotion_Label'] = df.apply(categorize_emotion, axis=1)
    return df, pa_cols, na_cols

def plot_matched_sample(edf_path, df_ratings, pa_cols, na_cols):
    """讀取腦波並附加對應的情緒標籤資訊後展示"""
    filename = os.path.basename(edf_path)
    print("\n" + "=" * 60)
    print(f"【檢視腦波與情緒對應】檔案: {filename}")

    # 1. 檢索情緒標籤
    matched_info = "無對應情緒標籤"
    detail_text = ""
    if df_ratings is not None:
        matched = df_ratings[df_ratings['Filename'] == filename]
        if not matched.empty:
            row = matched.iloc[0]
            matched_info = row['Emotion_Label']
            hi_pa = [col.replace("SR_", "") for col in pa_cols if row[col] >= 2]
            hi_na = [col.replace("SR_", "") for col in na_cols if row[col] >= 2]
            print(f"  [情緒標籤]: {matched_info}")
            print(f"  [得分詳情]: 正向(PA) = {row['PA_Score']} 分 | 負向(NA) = {row['NA_Score']} 分")
            print(f"  [報告字數]: {row['DreamReport_Wordcount']} 字")
            print(f"  [主要正向反應 (>=2分)]: {hi_pa if hi_pa else '無'}")
            print(f"  [主要負向反應 (>=2分)]: {hi_na if hi_na else '無'}")
            detail_text = f" | PA:{row['PA_Score']} NA:{row['NA_Score']} | {matched_info}"
        else:
            print("  [提示]: Ratings.csv 中無此檔評分（可能為無夢或無回報喚醒）")

    # 2. 讀取並展示腦波
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  [錯誤]: 腦波讀取失敗: {e}")
        return

    sampling_rate = raw.info['sfreq']
    n_channels = len(raw.ch_names)
    duration = raw.n_times / sampling_rate
    print(f"  [生理訊號]: 採樣率 {sampling_rate} Hz | 通道數 {n_channels} | 長度 {duration:.1f} 秒")
    print("=" * 60)

    # 顯示前 10 個核心導聯
    display_channels = raw.ch_names[:10]
    window_title = f"{filename}{detail_text}"

    raw.plot(
        duration=10.0,
        n_channels=len(display_channels),
        order=[raw.ch_names.index(ch) for ch in display_channels],
        scalings='auto',
        title=window_title,
        show=True,
        block=True
    )

def main():
    edf_files = get_edf_files()
    if not edf_files:
        print("未找到任何 .edf 檔案！請確認 Data/PSG 內是否有檔案。")
        return

    ratings_tuple = load_ratings()
    if ratings_tuple is None:
        return
    df_ratings, pa_cols, na_cols = ratings_tuple

    total = len(edf_files)
    current_idx = 0
    print(f"已準備就緒，共找到 {total} 個腦波檔案及情緒標籤表。")

    while True:
        current_file = edf_files[current_idx]
        print("\n" + "-" * 45)
        print(f"目前指針 [{current_idx + 1}/{total}]: {os.path.basename(current_file)}")
        print("操作指令:")
        print("  [Enter] 檢視當前樣本（腦波視窗 + 情緒標籤）")
        print("  [n] 下一個檔案並開啟")
        print("  [p] 上一個檔案並開啟")
        print(f"  [數字] 直接跳至指定編號 (1 ~ {total})")
        print("  [q] 離開程式")
        cmd = input("請輸入指令: ").strip().lower()

        if cmd == 'q':
            print("結束整合檢視。")
            break
        elif cmd == 'n':
            if current_idx + 1 < total:
                current_idx += 1
                plot_matched_sample(edf_files[current_idx], df_ratings, pa_cols, na_cols)
            else:
                print("已經是最後一個檔案了！")
        elif cmd == 'p':
            if current_idx - 1 >= 0:
                current_idx -= 1
                plot_matched_sample(edf_files[current_idx], df_ratings, pa_cols, na_cols)
            else:
                print("已經是第一個檔案了！")
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < total:
                current_idx = idx
                plot_matched_sample(edf_files[current_idx], df_ratings, pa_cols, na_cols)
            else:
                print(f"請輸入 1 到 {total} 之間的數字。")
        elif cmd == "":
            plot_matched_sample(current_file, df_ratings, pa_cols, na_cols)
        else:
            print("無法辨識的指令，請重新輸入。")

if __name__ == "__main__":
    main()