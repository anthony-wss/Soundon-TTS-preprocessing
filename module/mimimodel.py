from huggingface_hub import hf_hub_download
import torch

from moshi.models import loaders, LMGen
import numpy as np


"""
TODO: Add cuda support
"""
class MimiModel:
    def __init__(self):
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device='cpu')
        self.mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi.
        self.mimi.cuda()

    def pad_or_truncate(self, batch_audio, target_length):
        processed_audio = []
        for audio in batch_audio:
            if len(audio) < target_length:
                # Pad with zeros
                padded_audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                # Truncate to target length
                padded_audio = audio[:target_length]
            processed_audio.append(padded_audio)
        processed_audio = np.array(processed_audio)
        return torch.from_numpy(processed_audio).unsqueeze(1)


    def encode(self, audios, sample_rate):
        target_length = max(len(audio) for audio in audios)  # Example: Use the longest audio
        batch_audio_padded = self.pad_or_truncate(audios, target_length)
        with torch.no_grad():
            batch_audio_padded = batch_audio_padded.cuda()
            codes = self.mimi.encode(batch_audio_padded)

        return codes

