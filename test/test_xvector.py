from module.xvectormodel import XVectorModel
import torchaudio
import librosa
import torch


audio_files = ["/home/u3937558/ffa4ad00-3fc3-4f6d-a797-4cee575fce65-1.wav", "/work/u3937558/Soundon-TTS-preprocessing/output-0.wav"]
batch_audio = []
for file in audio_files:
    # audio, sample_rate = torchaudio.load(f"output-{i}.wav")
    audio, sample_rate = torchaudio.load(file)
    if sample_rate != 16000:
        audio = librosa.resample(audio.numpy(), orig_sr=sample_rate, target_sr=16000)
    audio = audio[0]
    batch_audio.append(audio)

cosine_sim = torch.nn.CosineSimilarity(dim=-1)
model = XVectorModel()
out = model.encode(batch_audio)
for i in range(out.shape[0]):
    for j in range(out.shape[0]):
        print(cosine_sim(out[i], out[j]), end="\t")
    print()
