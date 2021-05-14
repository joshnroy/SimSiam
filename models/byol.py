import copy
import random 
import torch 
from torch import nn 
import torch.nn.functional as F 
from torchvision import transforms 
from math import pi, cos 
from collections import OrderedDict
HPS = dict(
    max_steps=int(1000. * 1281167 / 4096), # 1000 epochs * 1281167 samples / batch size = 100 epochs * N of step/epoch
    # = total_epochs * len(dataloader) 
    mlp_hidden_size=512,
    projection_size=256,
    base_target_ema=5e-4,
    batchnorm_kwargs=dict(
        decay_rate=0.9,
        eps=1e-5), 
    seed=1337,
)


from .simsiam import D  # a bit different but it's essentially the same thing: neg cosine sim & stop gradient


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, HPS['mlp_hidden_size']),
            nn.BatchNorm1d(HPS['mlp_hidden_size'], eps=HPS['batchnorm_kwargs']['eps'], momentum=1-HPS['batchnorm_kwargs']['decay_rate']),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(HPS['mlp_hidden_size'], HPS['projection_size'])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projector = MLP(backbone.output_dim)
        self.online_encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        print("first#########################", flush=True)
        print(len([x for x in self.online_encoder.parameters()]), flush=True)
        print(len([x for x in self.backbone.parameters()]), flush=True)
        print("first_done#########################", flush=True)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_predictor = MLP(HPS['projection_size'])
        # raise NotImplementedError('Please put update_moving_average to training')

    def target_ema(self, k, K, base_ema=HPS['base_target_ema']):
        # tau_base = 0.996 
        # base_ema = 1 - tau_base = 0.996 
        # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2 

        tau_base = 1. - base_ema
        return 1. - (1. - tau_base) * (cos(pi*k/K)+1)/2 

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, HPS['max_steps'])
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
            
    def forward(self, x1, x2):
        f_o, h_o = self.online_encoder, self.online_predictor
        f_t      = self.target_encoder

        with torch.no_grad():
            print(len([x for x in f_o.parameters()]))
            total_diff = 0.
            for p1, p2 in zip(f_o.parameters(), f_t.parameters()):
                total_diff += torch.abs(p1.data - p2.data).sum().item()
            print("\n", total_diff, flush=True)

        z1_o = f_o(x1)
        z2_o = f_o(x2)

        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            z1_t = f_t(x1)
            z2_t = f_t(x2)
        
        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2 
        return {'loss': L}

    

if __name__ == "__main__":
    pass
