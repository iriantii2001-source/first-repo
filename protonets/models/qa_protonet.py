import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .few_shot import Flatten
from .utils import euclidean_dist

def minmax_normalize(x, dim, eps=1e-8):
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)

class QAProtonet(nn.Module):
    """Prototypical Network whose class prototypes are a *weighted* mean of the
    support embeddings instead of a simple mean. The weight of each support image
    is a fusion of (a) its no-reference image-quality score, precomputed offline
    with ARNIQA and passed in via sample['qs'], and (b) its typicality, i.e. how
    close its embedding is to the (unweighted) mean of its class in feature space."""

    def __init__(self, encoder, alpha=0.5, tau=1.0):
        super(QAProtonet, self).__init__()

        self.encoder = encoder
        self.alpha = alpha
        self.tau = tau

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        if 'qs' not in sample:
            raise KeyError("qa_protonet_conv requires per-support quality scores "
                            "(sample['qs']); re-run with --data.compute_quality")
        qs = sample['qs'] # n_class x n_support, ARNIQA score in [0, 1]

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        zs = z[:n_class * n_support].view(n_class, n_support, z_dim)
        zq = z[n_class * n_support:]

        # typicality: how close each support embedding is to its class's (unweighted) mean
        class_mean = zs.mean(1, keepdim=True) # n_class x 1 x z_dim
        typicality = -torch.norm(zs - class_mean, dim=-1) # n_class x n_support

        quality_norm = minmax_normalize(qs, dim=1)
        typicality_norm = minmax_normalize(typicality, dim=1)

        fused = self.alpha * quality_norm + (1 - self.alpha) * typicality_norm
        weights = F.softmax(fused / self.tau, dim=1) # n_class x n_support, sums to 1 per class

        z_proto = (zs * weights.unsqueeze(-1)).sum(1) # n_class x z_dim

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

@register_model('qa_protonet_conv')
def load_qa_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    alpha = kwargs.get('alpha', 0.5)
    tau = kwargs.get('tau', 1.0)

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return QAProtonet(encoder, alpha=alpha, tau=tau)
