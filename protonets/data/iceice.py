import os
import glob

from functools import partial

import torch
from torchvision.transforms import Compose, Resize, ToTensor

from protonets.data.base import (ListDataset, TransformDataset, compose, convert_dict,
                                  CudaTransform, EpisodicBatchSampler, SequentialBatchSampler)

ICEICE_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../dataset ice ice')
ICEICE_IMAGE_SIZE = 84
ICEICE_IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.webp')
ICEICE_SPLIT_DIRS = {
    'train': ['train'],
    'val': ['valid'],
    'test': ['test'],
    'trainval': ['train', 'valid'],
}
ICEICE_CACHE = { }

def get_class_dir_names():
    return sorted(d for d in os.listdir(ICEICE_DATA_DIR)
                  if os.path.isdir(os.path.join(ICEICE_DATA_DIR, d)))

def load_image_path(key, out_field, d):
    from PIL import Image
    d[out_field] = Image.open(d[key]).convert('RGB')
    return d

def convert_tensor(key, transform, d):
    d[key] = transform(d[key])
    return d

def load_class_images(d):
    if d['class'] not in ICEICE_CACHE:
        class_dir, split_name = d['class'].split('/')

        class_images = []
        for sub_dir in ICEICE_SPLIT_DIRS[split_name]:
            image_dir = os.path.join(ICEICE_DATA_DIR, class_dir, sub_dir)
            for ext in ICEICE_IMAGE_EXTS:
                class_images.extend(glob.glob(os.path.join(image_dir, ext)))
        class_images = sorted(class_images)

        if len(class_images) == 0:
            raise Exception("No images found for ice-ice class {} at {}".format(
                d['class'], os.path.join(ICEICE_DATA_DIR, class_dir)))

        transform = Compose([Resize((ICEICE_IMAGE_SIZE, ICEICE_IMAGE_SIZE)), ToTensor()])

        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             partial(convert_tensor, 'data', transform)]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            ICEICE_CACHE[d['class']] = sample['data']
            break # only need one sample because batch size equal to dataset length

    return { 'class': d['class'], 'data': ICEICE_CACHE[d['class']] }

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        transforms = [partial(convert_dict, 'class'),
                      load_class_images,
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = ["{:s}/{:s}".format(c, split) for c in get_class_dir_names()]
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), min(n_way, len(ds)), n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
