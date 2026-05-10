import torch.nn as nn
from torch.nn.functional import normalize
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, views, input_size, feature_dim):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.views = views
        for v in range(views):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.cluster_attn = nn.MultiheadAttention(feature_dim, num_heads=1, dropout=0.1)
        self.feature_attn = nn.MultiheadAttention(feature_dim, num_heads=1, dropout=0.1)

    def forward(self, xs):
        zs = []
        xrs = []
        for v in range(self.views):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            zs.append(z)
            xrs.append(xr)
        return zs, xrs

    def updateC(self, z, Pk, device):
        # feat_z = z.unsqueeze(0).permute(0, 2, 1)
        # feat_c = F.adaptive_avg_pool1d(feat_z, Pk)
        # feat_init = feat_c.squeeze(0).permute(1, 0)
        initial_prototypes = z[:Pk]

        max_iterations = 10
        tolerance = 1e-5
        for iteration in range(max_iterations):
            new_prototypes = self.cluster_attn(initial_prototypes, z, z)[0]
            diff = torch.norm(new_prototypes - initial_prototypes, dim=1).max().to(device)

            if max_iterations >= 10:
                initial_prototypes = z[:Pk]
            else:
                initial_prototypes = new_prototypes
            if diff < tolerance:
                break

        return initial_prototypes

    def imputeZ(self, z, Pa, device):
        feat_d = []
        for v in range(self.views):
            feat_d.append([])
        for v in range(self.views):
            feat_d[v] = self.feature_attn(z[v], Pa[v], Pa[v])[0]  # (N x C)
            feat_d[v] = feat_d[v] + z[v]
        return feat_d






