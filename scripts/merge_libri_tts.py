import torch
import librosa
import os
import soundfile as sf
from tqdm import tqdm


root_dir = "/home/anthony/disk-1t/"

if __name__ == "__main__":
    for ds_name in ["train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"]:
        os.makedirs(os.path.join("./libritts_pure_audio/", ds_name), exist_ok=True)
        progress = 0
        file_path_list = {}
        for root, dirs, files in os.walk(os.path.join(root_dir, ds_name)):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                file_path = os.path.join(root, file)
                spk_id = file_path.split('/')[-3]
                if spk_id not in file_path_list:
                    file_path_list[spk_id] = []
                file_path_list[spk_id].append(file_path)

        
        for spk_id in tqdm(file_path_list):
            audio_buffer = []
            dur = 0
            clip_id = 0
            os.makedirs(os.path.join("./libritts_pure_audio/", ds_name, spk_id), exist_ok=True)
            for file_path in file_path_list[spk_id]:
                # wav = audio_data_manager.get(file_path, 0, 16000)
                wav, sr = sf.read(file_path)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                dur += wav.shape[0] / 16000
                audio_buffer.append(torch.from_numpy(wav))

                if dur > 3600:
                    wav = torch.concat(audio_buffer)
                    sf.write(os.path.join("./libritts_pure_audio/", ds_name, spk_id, f"clip-{clip_id}.wav"), wav, 16000)
                    clip_id += 1
                    dur = 0
                    audio_buffer = []

            if dur > 0:
                wav = torch.concat(audio_buffer)
                sf.write(os.path.join("./libritts_pure_audio/", ds_name, spk_id, f"clip-{clip_id}.wav"), wav, 16000)

