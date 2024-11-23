from module.datasets.dataset import TTSDatasetLoader
import os


class LibriTTSLoader(TTSDatasetLoader):
    def __init__(self, root_dir, split_name):
        assert split_name in ["train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"]
        os.makedirs(os.path.join("./libritts_pure_audio/", split_name), exist_ok=True)
        self.data = []
        for root, dirs, files in os.walk(os.path.join(root_dir, split_name)):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                audio_file_path = os.path.join(root, file)
                text_file_path = audio_file_path[:-3] + 'csv'
                self.data.append([audio_file_path, text_file_path])
        
