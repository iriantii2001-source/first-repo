import os
import json

import torch
from PIL import Image
from torchvision.transforms import ToTensor

try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

_METRIC = None

def _get_metric():
    global _METRIC
    if _METRIC is None:
        import pyiqa
        _METRIC = pyiqa.create_metric('arniqa', device='cpu')
        _METRIC.eval()
    return _METRIC

def score_image(path):
    metric = _get_metric()
    img = Image.open(path).convert('RGB')
    t = ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        return float(metric(t).squeeze())

def load_scores(image_paths, cache_path):
    """Return a list of no-reference image-quality scores (ARNIQA, higher = better)
    aligned with image_paths, using a JSON cache keyed by path relative to the
    cache file's directory so newly added images only cost one extra score each."""
    cache = { }
    if os.path.isfile(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)

    cache_dir = os.path.dirname(cache_path)
    updated = False
    scores = []
    for path in image_paths:
        key = os.path.relpath(path, cache_dir)
        if key not in cache:
            cache[key] = score_image(path)
            updated = True
        scores.append(cache[key])

    if updated:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)

    return scores
