import whisperx
from datetime import timedelta
from tqdm import tqdm
from dotenv import load_dotenv
import os
import time
import gc
import torch
import logging


# ログ出力
logging.basicConfig(level=logging.INFO, filename="info.log",
                    format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

load_dotenv()
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")


# スピーカー数を入力
def get_speaker_count(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

MIN_SPEAKER = get_speaker_count("最少の参加者数を入力: ")
MAX_SPEAKER = get_speaker_count("最多の参加者数を入力: ")

# dataフォルダを作成
if not os.path.isdir("data"):
    os.mkdir("data")

if not os.path.isdir("model_dir"):
    os.mkdir("model_dir")


# フォルダ内のmp4ファイルをすべて取得
files = os.listdir("data")
video_files = []
for file in files:
    if file.endswith(".mp4"):
        video_files.append(os.path.join("data", file))


# モデルの読み込み
device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = (
    "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
)

# 1. Transcribe with original whisper (batched)
# model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
model_dir = "model_dir"
model = whisperx.load_model(
    "large-v3", device, compute_type=compute_type, download_root=model_dir
)


# 音声をテキストに変換
def transcribe_video(video_file, model, min_speakers=1, max_speakers=1):
    audio = whisperx.load_audio(video_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"])  # before alignment

    # delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    # print(result["segments"])  # after alignment

    # delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hugging_face_token, device=device
    )

    # add min/max number of speakers if known
    diarize_segments = diarize_model(video_file)
    diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"])  # segments are now assigned speaker IDs
    return result


# 結果をテキスト形式で保存
def save_result(result, video_file):
    with open(f"{video_file[:-4]}.txt", "w", encoding="utf-8") as f:
        for segment in tqdm(
            result["segments"], desc="Processing segments", ncols=75
        ):
            start_time = str(timedelta(seconds=segment["start"]))[:7]
            end_time = str(timedelta(seconds=segment["end"]))[:7]
            speaker = segment["speaker"]
            text = segment["text"]
            f.write(f"{start_time}-{end_time}\n{speaker}\n{text}\n\n")
            print(f"\n{start_time}-{end_time}\n{speaker}\n{text}\n")


# メイン関数
def main():
    start_time = time.time()
    for video_file in tqdm(video_files):
        result = transcribe_video(
            video_file,
            model,
            min_speakers=MIN_SPEAKER,
            max_speakers=MAX_SPEAKER,
        )
        save_result(result, video_file)
    logging.info("All done!")
    logging.info(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
