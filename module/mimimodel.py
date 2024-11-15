from transformers import AutoFeatureExtractor 
from transformers import MimiModel as hf_MimiModel
import torch


"""
TODO: Add cuda support
"""
class MimiModel:
    def __init__(self):
        model_id = "kyutai/mimi"
        self.model = hf_MimiModel.from_pretrained(model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model.cuda()

    def encode(self, wav, sample_rate):
        inputs = self.feature_extractor(raw_audio=wav, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move each input tensor to GPU
            outputs = self.model(**inputs)
        audio_codes = outputs.audio_codes
        return audio_codes

