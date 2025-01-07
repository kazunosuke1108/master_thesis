import requests
import json

# テキストをファイルに保存（必要に応じてスキップ可能）
text = "こんにちは、音声合成の世界へようこそ"
with open("text.txt", "w", encoding="utf-8") as f:
    f.write(text)

# 音声合成APIのエンドポイント
base_url = "http://127.0.0.1:50021"
speaker_id = 1

# 音声クエリの生成
query_endpoint = f"{base_url}/audio_query?speaker={speaker_id}"
response = requests.post(query_endpoint, params={"text": text})
if response.status_code != 200:
    print("音声クエリ生成に失敗しました:", response.text)
    exit()

# クエリデータをJSONとして保存
query_json = response.json()
with open("query.json", "w", encoding="utf-8") as f:
    json.dump(query_json, f, ensure_ascii=False, indent=4)

# speedScaleの値を変更
query_json["speedScale"] = 1.5

# 変更後のクエリデータを再保存
with open("query.json", "w", encoding="utf-8") as f:
    json.dump(query_json, f, ensure_ascii=False, indent=4)

# 音声合成リクエスト
synthesis_endpoint = f"{base_url}/synthesis?speaker={speaker_id}"
headers = {"Content-Type": "application/json"}
response = requests.post(synthesis_endpoint, headers=headers, json=query_json)
if response.status_code != 200:
    print("音声合成に失敗しました:", response.text)
    exit()

# 音声データを保存
with open("audio_fast.mp3", "wb") as f:
    f.write(response.content)

print("音声ファイルが生成されました: audio_fast.wav")

import sounddevice as sd
import scipy
import scipy.io.wavfile as wav
import soundfile as sf

# 再生するWAVファイル
wav_file = "audio_fast.wav"

data, sr = sf.read(wav_file)
print(f"Original sample rate: {sr}")

# WAVファイルを読み込み
sample_rate, data = wav.read(wav_file)

target_sr = 44100  # 変換後のサンプルレート
resample_factor = target_sr / sr
data_resampled = scipy.signal.resample(data, int(len(data) * resample_factor))

# 音声を再生
print(sample_rate)
sd.play(data_resampled, samplerate=target_sr)
sd.wait()  # 再生が終了するまで待機

print("再生が完了しました。")
