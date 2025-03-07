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
import playsound

# 音声合成サーバーのdocker: docker pull voicevox/voicevox_engine:cpu-latest
# dockerの起動：docker run --rm -p '127.0.0.1:50021:50021' voicevox/voicevox_engine:cpu-latest


class Notification(Manager):
    def __init__(self,trial_name="20250000Audio",strage="NASK"):
        # 音声合成APIのエンドポイント
        self.base_url = "http://127.0.0.1:50021"
        self.speaker_id = 23
        self.trial_name=trial_name
        self.strage=strage
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)
        pass

    def generate_audio(self,text,mp3_path,speed=1.25):
        query_endpoint = f"{self.base_url}/audio_query?speaker={self.speaker_id}"
        response = requests.post(query_endpoint, params={"text": text})
        if response.status_code != 200:
            print("音声クエリ生成に失敗しました:", response.text)
            exit()      

        # speedScaleの値を変更
        query_json = response.json()
        query_json["speedScale"] = speed

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
        if chime_type==0:
            combined_audio=AudioSegment.from_file(mp3_path)
        else:
            combined_audio = AudioSegment.from_file(chime_mp3_path)+AudioSegment.from_file(mp3_path)

        combined_audio.export(mp3_path, format="mp3")

        pass

    def play_only_chime(self,chime_type=1):
        chime_mp3_path=self.data_dir_dict["common_dir_path"]+f"/alert{chime_type}.mp3"
        playsound.playsound(chime_mp3_path, False)
    
    def play_mp3(self,mp3_path):
        playsound.playsound(mp3_path,False)

    def save_audio(self,audio_data):
        pass

    def export_audio(self,text,mp3_path,chime_type=1,speed=1.25):
        mp3_path=self.generate_audio(text=text,mp3_path=mp3_path,speed=speed)
        self.export_with_chime(mp3_path=mp3_path,chime_type=chime_type)

    def main_dev(self):
        # メッセージの生成
        # text="Aさんが，車椅子に乗っていて，ブレーキを解除しているので，危険です"
        # self.export_audio(text=text,mp3_path="/media/hayashide/MasterThesis/20250121VoiceDemo"+"/unbrake.mp3",chime_type=1,speed=1.25)
        # text="Cさんが，見守られていないので，危険です"
        # self.export_audio(text=text,mp3_path="/media/hayashide/MasterThesis/20250121VoiceDemo"+"/outofsight.mp3",chime_type=1,speed=1.25)
        # text="Aさんが，手すりから離れているのに，立ち上がろうとしているので，危険です"
        # self.export_audio(text=text,mp3_path="/media/hayashide/MasterThesis/20250121VoiceDemo"+"/standup.mp3",chime_type=1,speed=1.25)
        # text="複数の患者さんの対応が必要です．デイルームに来てください．"
        # self.export_audio(text=text,mp3_path="/media/hayashide/MasterThesis/20250121VoiceDemo"+"/helpus.mp3",chime_type=2,speed=1.25)
        # text="1.	序論 少子高齢化に伴う見守りを要する入院患者の増大が課題である．一方，身体拘束看護の廃止や医療介護人材不足の観点から医療従事者における見守りの負担は増大している．病棟内事故予防のため，危険な動作・状態にある患者の優先順位を評価し通知する見守り補佐システムが必要である．既存手法では，特定の危険動作や動作実行順の異常を患者毎に検出する例が多い．一方，同一動作でも本人の属性や空間的文脈に応じ危険度が異なることを考慮した例はなく，患者間の優先順位付けに基づく見守りは未実現である． 本研究では，病棟共有空間において危険な動作・状態にある患者を，空間的文脈を考慮した評価により検出し，医療従事者に通知するシステムを提案する．システムが現場の危険性評価基準に対し適応的であるために，評価の特徴量選定や推論に現場の不文律的知識を導入でき，調整も容易な体系を提案する．具体的には，多種の評価基準を反映するため階層化意思決定法を基盤とする．抽象的な評価則を反映するため，構造の多層化とFuzzy推論を導入する．患者の属性等の抽象的特徴量を実時間内に取得するため，マルチモーダル人工知能による特徴量取得と推論周期制御法を提案する．階層構造は評価結果の説明可能性にも貢献し，通知の具体性や医療従事者による基準調整の直感性に寄与する．提案手法はシミュレーションおよび実データへの適用により検証する．    Fig. 1 The situation where notification is required 2.	提案システム 機能フローをFig. 2に示す．システムは約20Hzで環境情報を取得し事前処理により特徴量を算出し，各患者について危険度を評価する．患者間の危険度の優先順位を基に適宜医療従事者への通知や応援要請を行う． 2.1.	観測・事前処理 まず，RGBカメラと3次元LiDARから環境画像と点群を取得し，各患者を抽出する．次に，事前処理として危険度評価に必要な特徴量を算出する．RGB画像からVisual Question Answering (VQA) により人物の属性・年齢層の特徴量を算出する．RGB画像への姿勢推定結果と現場知識で定義した危険動作との姿勢特徴の類似度から，動作特徴の特徴量を算出する．RGB画像へのVQAにより現場知識で定義した危険物体の有無を検出し，周辺物体の特徴量を算出する．患者と医療従事者の位置情報から相対距離と視認性の特徴量を算出する．VQAや姿勢推定に要する計算時間が大きくとも実時間評価を可能とするため，背景差分画像を用いて定量化した患者の動きの量等に応じた事前処理のフレームレート制御を導入する． Fig. 2  Function flow chart Fig. 3  Structure of the risk evaluation system 2.2.	危険度評価 手法の概要をFig. 3に示す．第3層では，患者の属性・年齢層の特徴量からファジィ積で内的・静的危険度を算出する．現場ヒアリングによる一対比較により，動作特徴量から内的・動的危険度を算出する．同様に，周辺物体の特徴量から外的・静的危険度を算出する．医療従事者との距離・視認性の大小に対し危険性が評価できるため， Fuzzy推論により外的・動的危険度を算出する．第2層以上では知識により各危険度の大小に対する危険性が定義可能なため，主にFuzzy推論で危険度を算出する． 2.3.	優先順位付け・通知生成 危険度上昇や順位変化に応じてFig. 4に従い通知を生成する．根拠明示のため，危険度算出時の階層構造の中位層の値に着目し，危険度が定常的に高い要因と現在危険度が上昇している要因を検出する仕組みを実現する． 3.	シミュレーション検証 実例と現場従事者へのヒアリングを基に作成した3名の患者が次々に立ち上がるシナリオについて，危険度の算出と通知文章の生成を検証した．立ち上がった人物に対して高い値を取る危険度が算出され，立ち上がりに関する状況説明を含む通知文が生成できた．また，対応が必要な患者数が担当可能スタッフ数を超越したことを検知し，他スタッフに応援を要請できたことを確認した．また，患者および見守り従事者の位置を網羅的に変更した9072例について検証し，ヒアリングで定義された方針に合致した危険度優先順位付けの実現を確認した．   Fig. 4 Notification algorithm 4.	実データ検証 病棟にセンサを設置し，許可のない立ち上がり・車椅子操作を記録した約30秒のデータ2例を提案手法で評価した．その結果，立ち上がる患者や車椅子を操作する患者を最優先と評価し，その内容を説明した通知文章を生成できた一方，微小な動作への過敏な反応や数秒おきの通知発報，画像平面上での隠れによる患者の取違等の課題も確認され，ノイズ耐性の改善が必要と考えられる． 5.	結論 本研究では病棟共有空間における空間的文脈を考慮した危険度モニタリングシステムを提案した．現場知識に基づく多種・高抽象度の評価基準を体系的に評価し，危険状態にある患者について根拠を含む通知文章の生成を確認した．実運用に向けて精度向上が課題となる． "         
        # self.export_audio(text=text,mp3_path="/media/hayashide/MasterThesis/20250121VoiceDemo"+"/abstract.mp3",chime_type=1,speed=1.25)
        # self.export_audio(text=text,mp3_path="/media/hayashide/MasterThesis/20250121VoiceDemo"+"/chime1.mp3",chime_type=1,speed=1.25)
        # self.export_audio(text=text,mp3_path="/media/hayashide/MasterThesis/20250121VoiceDemo"+"/chime2.mp3",chime_type=2,speed=1.25)
        self.data_dir_dict=self.get_database_dir(trial_name=self.trial_name,strage=self.strage)

        
        text="スタッフが見ていないDさんが，姿勢を崩しています．"
        self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/0_{text}.mp3")
        
        # text="スタッフが見ていないDさんが，姿勢を崩しています．"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/0_{text}.mp3")
        # text="スタッフが見ていないIさんが，バランスを崩しています．"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/1_{text}.mp3")
        
        # text="スタッフが見ていないDさんが，バランスを崩しています．"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/0_{text}.mp3")
        # text="スタッフが見ていないFさんが，立ち上がっています．"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/1_{text}.mp3")
        # text="スタッフが見ていないAさんが，姿勢を崩しています．"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/2_{text}.mp3")
        # text="スタッフが見ていないLさんが，姿勢を崩しています．"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/3_{text}.mp3")


        # text="Aさんが、車椅子を動かそうとしています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="車椅子を自立して操縦してはいけないAさんが、車椅子を動かそうとしています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="Aさんが、姿勢を崩しています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="転倒リスクが高いAさんが、姿勢を崩しています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="Aさんが、立とうとしています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="転倒リスクが高いAさんが、立とうとしています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="Aさんが、顔を触っています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="チューブを挿入しているAさんが、顔を触っています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="Aさんが、手を挙げています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        # text="重症度の高いAさんが、手を挙げています"
        # self.export_audio(text=text,mp3_path=self.data_dir_dict["trial_dir_path"]+f"/{text}.mp3")
        
if __name__=="__main__":
    # trial_name="20250121VoiceDemo"
    # strage="NASK"
    # cls=Notification(trial_name,strage)
    # cls.main_dev()

    trial_name="20250227Visualize5"
    strage="local"
    cls=Notification(trial_name,strage)
    # cls.play_only_chime()
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
