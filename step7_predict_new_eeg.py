#
# 測試 1：直接按 Enter
# 程式會自動隨機挑選一個腦波檔案進行端到端（End-to-End）特徵提取與預測，並輸出機率分佈與診斷。
# 測試 2：拖曳外部全新檔案
# 如果電腦有任何放在桌面或其他資料夾的 .edf 檔，可以直接把檔案拖曳進終端機中，按 Enter 即可直接預測！
# 測試 3：命令列參數執行
# 也可以使用：python step7_predict_new_eeg.py "Data/PSG/case100_s23.edf" 直接預測特定檔案。


import os
import sys
import glob
import joblib
import mne
import numpy as np
import pandas as pd

MODEL_FILE = "eeg_emotion_model.pkl"

def load_trained_model():
    """載入步驟五儲存的模型與特徵名稱清單"""
    if not os.path.exists(MODEL_FILE):
        print(f"錯誤：找不到模型檔案 '{MODEL_FILE}'！")
        print("請確認是否已經執行過 step5_train_model.py 產生模型。")
        return None, None
    data = joblib.load(MODEL_FILE)
    return data['model'], data['feature_names']

def extract_features_from_edf(edf_path, expected_features):
    """讀取單一 EDF 檔案並精準對齊模型所需的特徵維度"""
    print(f"\n[1/3] 正在讀取腦波檔案: {os.path.basename(edf_path)} ...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # 進行 1.0 ~ 45.0 Hz 帶通濾波
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)

    sampling_rate = raw.info['sfreq']
    duration = raw.n_times / sampling_rate
    print(f"      採樣頻率: {sampling_rate} Hz | 訊號總長: {duration:.1f} 秒 | 包含通道: {len(raw.ch_names)} 個")

    print("[2/3] 正在計算頻段功率譜 (PSD) 與額葉特徵 ...")
    # 濾除非 EEG 導聯
    eeg_channels = [ch for ch in raw.ch_names if not any(x in ch.upper() for x in ['EOG', 'EMG', 'ECG', 'STATUS'])]

    spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=45.0, picks=eeg_channels, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)

    bands = {
        'Delta': (1.0, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 12.0),
        'Beta': (12.0, 30.0),
        'Gamma': (30.0, 45.0)
    }

    feature_dict = {}
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power = psds[:, idx_band].mean(axis=1)
        for ch_idx, ch_name in enumerate(eeg_channels):
            feature_dict[f"{ch_name}_{band_name}"] = band_power[ch_idx]

    # 計算 FAA (若無 F4/F3 則填 0)
    if 'F4' in eeg_channels and 'F3' in eeg_channels:
        f4_a = feature_dict.get('F4_Alpha', 1e-6)
        f3_a = feature_dict.get('F3_Alpha', 1e-6)
        feature_dict['FAA_F4_F3'] = np.log(max(f4_a, 1e-9)) - np.log(max(f3_a, 1e-9))
    else:
        feature_dict['FAA_F4_F3'] = 0.0

    # 依照模型訓練時的特徵順序精準對齊，缺少的通道補 0
    feature_vector = []
    missing_channels = []
    for f_name in expected_features:
        if f_name in feature_dict:
            feature_vector.append(feature_dict[f_name])
        else:
            feature_vector.append(1e-13)  # 給予合理基準值
            missing_channels.append(f_name)

    if missing_channels:
        print(f"      注意：該 EDF 檔缺少 {len(set([c.split('_')[0] for c in missing_channels]))} 個通道，已自動補齊基準值。")

    return np.array(feature_vector).reshape(1, -1), feature_dict

def predict_eeg_file(edf_path, model, feature_names):
    """輸入 EDF 檔案路徑並印出預測結果"""
    if not os.path.exists(edf_path):
        print(f"錯誤：找不到檔案 '{edf_path}'")
        return

    feat_vec, feat_dict = extract_features_from_edf(edf_path, feature_names)

    print("[3/3] 正在進行機器學習情緒推論 ...")
    pred_class_idx = model.predict(feat_vec)[0]
    pred_probs = model.predict_proba(feat_vec)[0]  # [Prob(Negative), Prob(Positive)]

    prob_neg = pred_probs[0] * 100
    prob_pos = pred_probs[1] * 100

    label_str = "Positive (正向情緒夢境)" if pred_class_idx == 1 else "Negative (負向情緒夢境)"
    confidence = prob_pos if pred_class_idx == 1 else prob_neg

    print("\n" + "=" * 65)
    print(f"【腦波情緒預測報告】: {os.path.basename(edf_path)}")
    print("=" * 65)
    print(f"  ★ 預測夢境情緒類別 : {label_str}")
    print(f"  ★ 模型判定信心度   : {confidence:.2f}%")
    print(f"  ★ 類別機率分佈     : 正向 (Positive) {prob_pos:.1f}%  |  負向 (Negative) {prob_neg:.1f}%")
    print("-" * 65)
    print("  [生理指標診斷]:")
    
    # 提取額葉關鍵指標進行診斷
    f7_theta = feat_dict.get('F7_Theta', None)
    af3_theta = feat_dict.get('AF3_Theta', None)
    fp2_gamma = feat_dict.get('Fp2_Gamma', None)
    
    if f7_theta and af3_theta:
        print(f"  - 前額葉 Theta 能量 (F7/AF3) : {f7_theta:.2e} / {af3_theta:.2e}")
    if fp2_gamma:
        print(f"  - 額極皮質高頻放電 (Fp2_Gamma): {fp2_gamma:.2e}")
        
    if pred_class_idx == 1:
        print("  - 特徵解讀: 前額葉慢波與高頻激發平穩，大腦無強烈警報放電反應。")
    else:
        print("  - 特徵解讀: 前額葉出現顯著激發放電，顯示做夢時存在較高焦慮或緊繃情緒。")
    print("=" * 65 + "\n")

def main():
    print("=" * 65)
    print("【步驟七：全新外部腦波即時情緒預測器】")
    print("=" * 65)

    model, feature_names = load_trained_model()
    if model is None:
        return

    # 支援在終端機直接傳遞檔名：python step7_predict_new_eeg.py <路徑>
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        predict_eeg_file(target_path, model, feature_names)
        return

    # 互動模式：讓使用者自行貼入路徑或檔名
    print("模式提示：你可以直接輸入檔案路徑，或輸入編號測試既有資料夾中的檔案。")
    all_edfs = sorted(glob.glob(os.path.join(".", "**", "*.edf"), recursive=True))

    while True:
        user_input = input("請輸入 EDF 檔案路徑 (或按 [Enter] 隨機挑選一筆測試，輸入 'q' 離開): ").strip()
        if user_input.lower() == 'q':
            print("退出預測器。")
            break

        if user_input == "":
            if all_edfs:
                selected = np.random.choice(all_edfs)
                print(f"隨機選取檔案: {selected}")
                predict_eeg_file(selected, model, feature_names)
            else:
                print("找不到任何 EDF 檔案，請手動輸入檔案完整路徑。")
        else:
            # 清理引號（方便使用者直接從檔案總管拖曳路徑進來）
            clean_path = user_input.strip("\"'")
            predict_eeg_file(clean_path, model, feature_names)

if __name__ == "__main__":
    main()