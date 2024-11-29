from torch.utils.data import DataLoader, Dataset
import random
import torch
import copy
import logging
from .utils import loading_pipeline

logger = logging.getLogger(__name__)

# def collate_fn(batch):
#     new_data = {'video': [], 'labels': []}
#     for data in batch:
#         new_data['video'].append(data['video'])
#         new_data['labels'].append(data['labels'])
#     new_data['video'] = torch.concatenate(new_data['video'],axis=0)
#     new_data['labels'] = torch.stack(new_data['labels'])
#     return new_data


class MyDataset(Dataset):
    def __init__(self,video_paths, labels=None,shuffle=False):
        if labels is None:
            labels = [0] * len(video_paths) # dummy
        self.video_paths, self.labels = video_paths, labels
        if shuffle:
            self.shuffle()
    
    def __len__(self):
        return len(self.video_paths)
    
    def shuffle(self):
        c = list(zip(self.video_paths, self.labels))
        random.shuffle(c)
        self.video_paths, self.labels = zip(*c)
    
    def __getitem__(self, idx):
        path, label = self.video_paths[idx], self.labels[idx]
        video = loading_pipeline(path)
        labels = torch.tensor(label).to('cpu')
        logger.info(f"Loaded video {path} with label {label}")
        return {'video': video, 'labels': labels}
    
    def _slice(self,start,end):
        self_copy = copy.deepcopy(self)
        self_copy.video_paths = self.video_paths[start:end]
        self_copy.labels = self.labels[start:end]
        return self_copy