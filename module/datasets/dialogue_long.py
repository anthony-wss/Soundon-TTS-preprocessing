from module.datasets.dataset import TTSDatasetLoader
import os


class DialogueLongTTSLoader(TTSDatasetLoader):
    def __init__(self, root_dir):
        self.data = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                audio_file_path = os.path.join(root, file)
                audio_id = audio_file_path.split('/')[-1][:-4]
                text_file_path = "/".join(audio_file_path.split('/')[:-2]) + f'/timestamp/{audio_id}.csv'
                self.data.append([audio_file_path, text_file_path])
 
