import os
import sys
sys.path.append(".")
sys.path.append("..")
if "catkin_ws" in os.getcwd():
    sys.path.append("/catkin_ws/src/master_thesis_modules")
else:
    sys.path.append(os.path.expanduser("~")+"/kazu_ws/master_thesis/master_thesis_modules")

import requests
import json

from scripts.management.manager import Manager

from pydub import AudioSegment

# 音声合成サーバーのdocker: docker pull voicevox/voicevox_engine:cpu-latest
# dockerの起動：docker run --rm -p '127.0.0.1:50021:50021' voicevox/voicevox_engine:cpu-latest


class Notification(Manager):
    def __init__(self,trial_name,strage):
        # 音声合成APIのエンドポイント
        self.base_url = "http://127.0.0.1:50021"
        self.speaker_id = 0
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)
        pass

    def generate_audio(self,text,mp3_path):
        query_endpoint = f"{self.base_url}/audio_query?speaker={self.speaker_id}"
        response = requests.post(query_endpoint, params={"text": text})
        if response.status_code != 200:
            print("音声クエリ生成に失敗しました:", response.text)
            exit()      

        # speedScaleの値を変更
        query_json = response.json()
        query_json["speedScale"] = 1.25

        # 変更後のクエリデータを再保存
        with open("query.json", "w", encoding="utf-8") as f:
            json.dump(query_json, f, ensure_ascii=False, indent=4)

        # 音声合成リクエスト
        synthesis_endpoint = f"{self.base_url}/synthesis?speaker={self.speaker_id}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(synthesis_endpoint, headers=headers, json=query_json)
        if response.status_code != 200:
            print("音声合成に失敗しました:", response.text)
            exit()
        # mp3_path=self.data_dir_dict["trial_dir_path"]+"/notification_text.mp3"
        with open(mp3_path, "wb") as f:
            f.write(response.content)        
        return mp3_path
    
        

    def export_with_chime(self,mp3_path,chime_type=1):
        chime_mp3_path=self.data_dir_dict["common_dir_path"]+f"/alert{chime_type}.mp3"
        combined_audio = AudioSegment.from_file(chime_mp3_path)+AudioSegment.from_file(mp3_path)
        combined_audio.export(mp3_path, format="mp3")

        pass

    def save_audio(self,audio_data):
        pass

    def export_audio(self,text,mp3_path,chime_type=1,):
        mp3_path=self.generate_audio(text=text,mp3_path=mp3_path)
        self.export_with_chime(mp3_path=mp3_path,chime_type=chime_type)

    def main_dev(self,text):
        # メッセージの生成
        text="患者さんが立ち上がりました。近くに車椅子があります。"
        mp3_path=self.generate_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+"/notification.mp3")
        self.export_with_chime(mp3_path,chime_type=1)
        text="デイルームで見守りのお手伝いをお願いします。"
        mp3_path=self.generate_audio(text=text)
        self.export_with_chime(mp3_path,chime_type=2)


        pass

if __name__=="__main__":
    trial_name="20250107VoiceNotification"
    strage="NASK"
    cls=Notification(trial_name,strage)
    cls.main_dev()


"""
四国めたん（ノーマル）: 2
四国めたん（あまあま）: 0
四国めたん（ツンツン）: 6
四国めたん（セクシー）: 4
四国めたん（ささやき）: 36
四国めたん（ヒソヒソ）: 37
ずんだもん（ノーマル）: 3
ずんだもん（あまあま）: 1
ずんだもん（ツンツン）: 7
ずんだもん（セクシー）: 5
ずんだもん（ささやき）: 22
ずんだもん（ヒソヒソ）: 38
春日部つむぎ（ノーマル）: 8
雨晴はう（ノーマル）: 10
波音リツ（ノーマル）: 9
玄野武宏（ノーマル）: 11
玄野武宏（喜び）: 39
玄野武宏（ツンギレ）: 40
玄野武宏（悲しみ）: 41
白上虎太郎（ふつう）: 12
白上虎太郎（わーい）: 32
白上虎太郎（びくびく）: 33
白上虎太郎（おこ）: 34
白上虎太郎（びえーん）: 35
青山龍星（ノーマル）: 13
冥鳴ひまり（ノーマル）: 14
九州そら（ノーマル）: 16
九州そら（あまあま）: 15
九州そら（ツンツン）: 18
九州そら（セクシー）: 17
九州そら（ささやき）: 19
もち子さん（ノーマル）: 20
剣崎雌雄（ノーマル）: 21
WhiteCUL（ノーマル）: 23
WhiteCUL（たのしい）: 24
WhiteCUL（かなしい）: 25
WhiteCUL（びえーん）: 26
後鬼（人間ver.）: 27
後鬼（ぬいぐるみver.）: 28
No.7（ノーマル）: 29
No.7（アナウンス）: 30
No.7（読み聞かせ）: 31
ちび式じい（ノーマル）: 42
櫻歌ミコ（ノーマル）: 43
櫻歌ミコ（第二形態）: 44
櫻歌ミコ（ロリ）: 45
小夜/SAYO（ノーマル）: 46
ナースロボ＿タイプＴ（ノーマル）: 47
ナースロボ＿タイプＴ（楽々）: 48
ナースロボ＿タイプＴ（恐怖）: 49
ナースロボ＿タイプＴ（内緒話）: 50
"""
# # テキストをファイルに保存（必要に応じてスキップ可能）
# text = "こんにちは、音声合成の世界へようこそ"
# with open("text.txt", "w", encoding="utf-8") as f:
#     f.write(text)

# # 音声合成APIのエンドポイント
# base_url = "http://127.0.0.1:50021"
# speaker_id = 1

# # 音声クエリの生成
# query_endpoint = f"{base_url}/audio_query?speaker={speaker_id}"
# response = requests.post(query_endpoint, params={"text": text})
# if response.status_code != 200:
#     print("音声クエリ生成に失敗しました:", response.text)
#     exit()

# # クエリデータをJSONとして保存
# query_json = response.json()
# with open("query.json", "w", encoding="utf-8") as f:
#     json.dump(query_json, f, ensure_ascii=False, indent=4)

# # speedScaleの値を変更
# query_json["speedScale"] = 1.5

# # 変更後のクエリデータを再保存
# with open("query.json", "w", encoding="utf-8") as f:
#     json.dump(query_json, f, ensure_ascii=False, indent=4)

# # 音声合成リクエスト
# synthesis_endpoint = f"{base_url}/synthesis?speaker={speaker_id}"
# headers = {"Content-Type": "application/json"}
# response = requests.post(synthesis_endpoint, headers=headers, json=query_json)
# if response.status_code != 200:
#     print("音声合成に失敗しました:", response.text)
#     exit()

# # 音声データを保存
# with open("audio_fast.mp3", "wb") as f:
#     f.write(response.content)

# print("音声ファイルが生成されました: audio_fast.wav")

# import sounddevice as sd
# import scipy
# import scipy.io.wavfile as wav
# import soundfile as sf

# # 再生するWAVファイル
# wav_file = "audio_fast.wav"

# data, sr = sf.read(wav_file)
# print(f"Original sample rate: {sr}")

# # WAVファイルを読み込み
# sample_rate, data = wav.read(wav_file)

# target_sr = 44100  # 変換後のサンプルレート
# resample_factor = target_sr / sr
# data_resampled = scipy.signal.resample(data, int(len(data) * resample_factor))

# # 音声を再生
# print(sample_rate)
# sd.play(data_resampled, samplerate=target_sr)
# sd.wait()  # 再生が終了するまで待機

# print("再生が完了しました。")
