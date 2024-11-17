import torch
from speechbrain.inference.speaker import EncoderClassifier

XVECTOR_SR = 16000

class XVectorModel:
    def __init__(self):
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})

    def encode(self, audios):
        for i in range(len(audios)):
            audios[i] = audios[i][:5*XVECTOR_SR]

        max_len = max([audio.shape[0] for audio in audios])
        wav_lens = torch.tensor([audio.shape[0]/max_len for audio in audios])

        padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0)
        embeddings = self.classifier.encode_batch(padded, wav_lens=wav_lens).cpu()
        return embeddings

