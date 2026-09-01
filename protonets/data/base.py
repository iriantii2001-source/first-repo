import torch

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, elem_list, load=lambda x: x):
        self.elem_list = elem_list
        self.load = load

    def __len__(self):
        return len(self.elem_list)

    def __getitem__(self, idx):
        return self.load(self.elem_list[idx])

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

def compose(transforms):
    def composed(x):
        for transform in transforms:
            x = transform(x)
        return x
    return composed

def convert_dict(k, v):
    return { k: v }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data

class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
