import torch
import numpy as np
from transformers import MimiModel as hf_MimiModel
from transformers import AutoFeatureExtractor
from math import ceil


class MimiModel:
    def __init__(self):
        self.mimi = hf_MimiModel.from_pretrained("kyutai/mimi")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
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


    def encode(self, audios):
        """
        audios: List[np.array]
        """
        if len(audios) == 1:
            inputs = self.feature_extractor(audios[0], sampling_rate=24000, return_tensors="pt")
            with torch.no_grad():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move each input tensor to GPU
                batch_output = self.mimi.encode(**inputs, num_quantizers=8).audio_codes.cpu()
            return batch_output

        # target_length = max(len(audio) for audio in audios)  # Example: Use the longest audio
        # batch_audio_padded = self.pad_or_truncate(audios, target_length)
        length = [ceil(audio.shape[0]/24000*12.5) for audio in audios]
        inputs = self.feature_extractor(audios, sampling_rate=24000, return_tensors="pt", padding=True)
        with torch.no_grad():
            # batch_audio_padded = batch_audio_padded.cuda()
            inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move each input tensor to GPU
            batch_output = self.mimi.encode(**inputs, num_quantizers=8).audio_codes.cpu()

        codes = []
        for i in range(batch_output.shape[0]):
            code_seq = batch_output[i][:, :length[i]]
            codes.append(code_seq)

        return codes

    def decode(self, unit):
        """ 
            Batch decode not supported yet
            unit: torch.tensor if shape [8, len]
        """
        assert len(unit.shape) == 2
        assert unit.shape[0] == 8
        unit = unit.unsqueeze(0)
        unit = unit.cuda()
        audio_values = self.mimi.decode(unit).audio_values.cpu().detach()
        return audio_values

