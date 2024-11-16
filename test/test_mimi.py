import torchaudio
import librosa
from module.mimimodel import MimiModel


batch_audio = []
for i in range(1):
    # audio, sample_rate = torchaudio.load(f"output-{i}.wav")
    audio, sample_rate = torchaudio.load("/home/u3937558/ffa4ad00-3fc3-4f6d-a797-4cee575fce65-1.wav")
    if sample_rate != 24000:
        audio = librosa.resample(audio.numpy(), orig_sr=sample_rate, target_sr=24000)
    audio = audio[0]
    batch_audio.append(audio)

model = MimiModel()
out = model.encode(batch_audio, 24000)
print([x.shape for x in out])


