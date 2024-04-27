# speech-to-text

文字起こし whisper に話者分離を追加した whisperX で処理

## whisperX

モデルのセットアップ方法<br>
https://github.com/m-bain/whisperX<br>
単語レベルのタイムスタンプと話者ダイアリゼーションによる高速自動音声認識<br>

## whisper モデル

"large-v3"モデル（2023 年 11 月～）を利用
https://huggingface.co/openai/whisper-large-v3

## CPU 利用の場合(main.py を修正)

evice="cpu"
compute_type = "int8"

## 話者分離

.env.sample ファイルを.env ファイルに書き換える
https://huggingface.co/settings/tokens　に登録してトークン取得
HUGGING_FACE_TOKEN=トークンを入力

## 話者の数を入力

最少の参加者: 最少の時の人数を入力(1 人～)<br>
最多の参加者: 最多の時の人数を入力<br>

## 入力する動画ファイル

data フォルダにアップロード(mp4 等)<br>

## 出力される文字起こし

data フォルダに text ファイルを出力<br>

## 出力されるログ

動画の長さと処理時間を出力<br>
