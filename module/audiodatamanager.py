import torchaudio


class AudioDataManager:
    """
    Save audio tensor on cpu.
    self.audio_cache = {
        "audio_file": {
            sampling_rate: array,
            sampling_rate: array
        }
    }
    """
    def __init__(self):
        self.audio_cache = {}
        self._reframe_list = {}

    def pop(self, audio_file):
        del self.audio_cache[audio_file]

    def get(self, audio_file, channel, sample_rate):
        # Load audio from disk
        if audio_file not in self.audio_cache:
            audio, sample_rate = torchaudio.load(audio_file)
            self.audio_cache[audio_file] = {sample_rate: audio}

        if sample_rate in self.audio_cache[audio_file]:
            return self.audio_cache[audio_file][sample_rate][channel]

        # Resample
        orig_rate = int(list(self.audio_cache[audio_file].keys())[0])
        audio = self.audio_cache[audio_file][orig_rate]
        resample_key = f"{orig_rate}:{sample_rate}"
        if resample_key not in self._reframe_list:
            self._reframe_list[resample_key] = torchaudio.transforms.Resample(
                orig_freq=orig_rate,
                new_freq=sample_rate
            ).to('cuda')
        audio = audio.cuda()
        audio = self._reframe_list[resample_key](audio)
        audio = audio.cpu()
        self.audio_cache[audio_file][sample_rate] = audio

        return audio[channel]

