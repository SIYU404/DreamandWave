import os
import glob
import mne
import numpy as np
import pandas as pd

def find_file(filename):
    """優先檢查 Data/，否則搜尋當前目錄與所有子目錄"""
    direct_path = os.path.join(".", "Data", filename)
    if os.path.exists(direct_path):
        return direct_path
    matches = glob.glob(os.path.join(".", "**", filename), recursive=True)
    return matches[0] if matches else filename

def get_edf_files():
    """搜尋所有 .edf 檔案"""
    files = glob.glob(os.path.join(".", "**", "*.edf"), recursive=True)
    return sorted(files)

def load_labels():
    """載入情緒問卷標籤"""
    ratings_path = find_file("Ratings.csv")
    if not os.path.exists(ratings_path):
        print(f"錯誤：找不到標籤檔案 {ratings_path}")
        return None

    df = pd.read_csv(ratings_path)
    pa_cols = [f"SR_PA{i}" for i in range(1, 11)]
    na_cols = [f"SR_NA{i}" for i in range(1, 11)]

    df['PA_Score'] = df[pa_cols].sum(axis=1)
    df['NA_Score'] = df[na_cols].sum(axis=1)
    df['Net_Valence'] = df['PA_Score'] - df['NA_Score']

    def assign_class(row):
        if row['PA_Score'] > row['NA_Score']:
            return "Positive"
        elif row['NA_Score'] > row['PA_Score']:
            return "Negative"
        else:
            return "Neutral"

    df['Emotion_Class'] = df.apply(assign_class, axis=1)
    # 建立檔名對應字典
    label_dict = {}
    for _, r in df.iterrows():
        fname = os.path.basename(r['Filename'])
        if not fname.endswith('.edf'):
            fname += '.edf'
        label_dict[fname] = {
            'PA_Score': r['PA_Score'],
            'NA_Score': r['NA_Score'],
            'Net_Valence': r['Net_Valence'],
            'Emotion_Class': r['Emotion_Class']
        }
    return label_dict

def extract_features_from_raw(raw):
    """計算腦波 5 大頻段功率 (PSD) 與額葉不對稱性 (FAA)"""
    # 濾除眼電(EOG)與肌電(EMG)，只保留頭皮 EEG 導聯
    eeg_channels = [ch for ch in raw.ch_names if not any(x in ch.upper() for x in ['EOG', 'EMG', 'ECG', 'STATUS'])]
    
    # 計算 Welch 功率譜密度 (0.5 ~ 45 Hz)
    spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=45.0, picks=eeg_channels, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # 定義五大頻段
    bands = {
        'Delta': (1.0, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 12.0),
        'Beta': (12.0, 30.0),
        'Gamma': (30.0, 45.0)
    }

    features = {}

    # 1. 提取各通道在各頻段的絕對功率
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # 平均頻譜功率
        band_power = psds[:, idx_band].mean(axis=1)
        for ch_idx, ch_name in enumerate(eeg_channels):
            features[f"{ch_name}_{band_name}"] = band_power[ch_idx]

    # 2. 計算額葉 Alpha 不對稱性 (FAA: ln(Right) - ln(Left))
    # Sikka 2019 論文的核心特徵：F4(右額) vs F3(左額)
    if 'F4' in eeg_channels and 'F3' in eeg_channels:
        f4_alpha = features.get('F4_Alpha', 1e-6)
        f3_alpha = features.get('F3_Alpha', 1e-6)
        features['FAA_F4_F3'] = np.log(max(f4_alpha, 1e-9)) - np.log(max(f3_alpha, 1e-9))

    return features

def main():
    print("=" * 60)
    print("【步驟四：腦波頻段特徵提取模組】")
    print("=" * 60)

    edf_files = get_edf_files()
    if not edf_files:
        print("未找到任何 .edf 檔案！請確認 Data/PSG 路徑。")
        return

    label_dict = load_labels()
    if label_dict is None:
        return

    print(f"找到 {len(edf_files)} 個腦波檔，正在比對情緒標籤並提取頻譜特徵...")

    extracted_rows = []
    matched_count = 0

    for i, file_path in enumerate(edf_files, 1):
        filename = os.path.basename(file_path)
        
        # 只提取有標籤對應的檔案
        if filename not in label_dict:
            continue

        try:
            # 讀取 EDF，自動進行 1 ~ 45 Hz 帶通濾波以濾除慢漂移
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
            
            features = extract_features_from_raw(raw)
            features['Filename'] = filename
            
            # 加入標籤
            features.update(label_dict[filename])
            extracted_rows.append(features)
            matched_count += 1
            
            if matched_count % 10 == 0 or i == len(edf_files):
                print(f"  已處理進度: [{i}/{len(edf_files)}] (成功萃取 {matched_count} 筆樣本特徵)")
        except Exception as e:
            print(f"  處理 {filename} 時出錯: {e}")

    # 輸出為特徵 CSV 檔
    output_df = pd.DataFrame(extracted_rows)
    output_filename = "eeg_features_dataset.csv"
    output_df.to_csv(output_filename, index=False)

    print("\n" + "=" * 60)
    print(f"特徵提取全部完成！")
    print(f"共萃取 {len(output_df)} 筆有效樣本，每筆包含 {len(output_df.columns) - 5} 個腦電特徵。")
    print(f"已儲存為結構化特徵集: {os.path.abspath(output_filename)}")
    print("情緒類別統計:")
    print(output_df['Emotion_Class'].value_counts().to_string())
    print("=" * 60)

if __name__ == "__main__":
    main()