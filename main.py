from argparse import ArgumentParser
import os
from multiprocessing import Process, Queue, set_start_method
import queue
from module.worker import PreprocessWorker
from module.datasets.dialogue_long import DialogueLongTTSLoader
import time
import torch
from datasets import Dataset, load_from_disk, concatenate_datasets
import logging

CONV_BATCH_SIZE = 3


def worker_func(worker_id, audio_queue, save_path):
    torch.cuda.set_device(worker_id)
    worker = PreprocessWorker()
    dataset_dict = {"machine_unit": [], "x-vector": [], "text": [], "text_with_pad": [], "user_audio_path": []}

    logger = logging.getLogger(f"worker_{worker_id}")
    handler = logging.FileHandler(f'example.{worker_id}.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    while True:
        try:
            time.sleep(.5)
            mini_batch = audio_queue.get(True, 60)
            samples_dict = worker.run_conv_batch(mini_batch)
            if samples_dict:
                logger.debug(f"Processing sample: {[x[0] for x in mini_batch]}")
                length = sum([len(x) for x in samples_dict["text"]])
                logger.debug(f"Content length: {length}")
                for key in dataset_dict.keys():
                    dataset_dict[key].extend(samples_dict[key])
            print("Queue size:", audio_queue.qsize())
            for audio_file in mini_batch:
                worker.audio_data_manager.pop(audio_file[0])
        except queue.Empty:
            logger.info("Queue is empty, worker is exiting.")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
    ds = Dataset.from_dict(dataset_dict) 
    ds.save_to_disk(os.path.join(save_path, str(worker_id)))

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    set_start_method("spawn", force=True)

    parser = ArgumentParser()

    # TODO: Make Soundon works by defining `datasets/soundon.py`
    parser.add_argument("--root_dir", required=True, help="Path to root dir containing audios/ and timestamp/")
    # parser.add_argument("--split_name", required=True)
    parser.add_argument("--num_worker", type=int, default=1)
    args = parser.parse_args()


    dataset = DialogueLongTTSLoader(root_dir=args.root_dir)
    save_path = os.path.join(".", args.root_dir.split('/')[-1])

    audio_queue = Queue()
    mini_batch = []
    for sample in dataset.iterate():
        mini_batch.append(sample)
        if len(mini_batch) >= CONV_BATCH_SIZE:
            audio_queue.put(mini_batch)
            mini_batch = []
    if mini_batch:
        audio_queue.put(mini_batch)

    number_of_processes = args.num_worker
    processes = []
    for w in range(number_of_processes):
        p = Process(target=worker_func, args=(w, audio_queue, save_path))
        processes.append(p)
        p.start()

    # Completing process
    for p in processes:
        p.join()

    whole_dataset = None
    for i in range(number_of_processes):
        ds = load_from_disk(os.path.join(save_path, str(i)))

        if not whole_dataset:
            whole_dataset = ds
        else:
            whole_dataset = concatenate_datasets([whole_dataset, ds])
    save_path = os.path.join(".", "hf_"+args.root_dir.split('/')[-1])
    whole_dataset.save_to_disk(save_path)

