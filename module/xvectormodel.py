import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertForXVector


class XVectorModel:
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.model = Wav2Vec2BertForXVector.from_pretrained("facebook/w2v-bert-2.0")
        self.model.cuda()

    def encode(self, audios, sample_rate):
        inputs = self.feature_extractor(audios, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move each input tensor to GPU
            embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        return embeddings

