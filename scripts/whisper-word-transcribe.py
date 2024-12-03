"""

Format:
speaker,start,end,text
0,0.00s,0.42s,朗
...

"""
import os
import csv
from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm import tqdm

ds_name = "train-clean-100"
root_dir = "/mnt/home/ntuspeechlabtaipei1/anthony/Soundon-TTS-preprocessing/libritts_pure_audio"


class BatchWhisper:
    def __init__(self):
        model_size = "large-v3"
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.batched_model = BatchedInferencePipeline(model=model)

    def transcribe(self, audio_path):
        segments, info = self.batched_model.transcribe(audio_path, batch_size=16, word_timestamps=True)
        segments = list(segments)
        return segments


if __name__ == "__main__":
    
    batch_whisper = BatchWhisper()

    for ds_name in ["train-clean-100", "train-clean-360"]:
        file_path_list = []
        for root, dirs, files in os.walk(os.path.join(root_dir, ds_name)):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                file_path = os.path.join(root, file)
                file_path_list.append(file_path)

        for file_path in tqdm(file_path_list):
            print("start")
            results = batch_whisper.transcribe(file_path)
        
            csv_file_path = file_path[:-3] + "csv"
            with open(csv_file_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for segment in results:
                    for chunk in segment.words:
                        writer.writerow([0, chunk.start, chunk.end, chunk.word])

            print(csv_file_path)

