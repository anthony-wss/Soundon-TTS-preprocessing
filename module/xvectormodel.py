import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertForXVector

XVECTOR_SR = 16000

class XVectorModel:
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.model = Wav2Vec2BertForXVector.from_pretrained("facebook/w2v-bert-2.0")
        self.model.cuda()

    def encode(self, audios):
        for i in range(len(audios)):
            audios[i] = audios[i][:5*XVECTOR_SR]
        inputs = self.feature_extractor(audios, sampling_rate=XVECTOR_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move each input tensor to GPU
            embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        return embeddings

