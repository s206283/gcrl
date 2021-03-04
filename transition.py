import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transition(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim, trans_type='linear'):
        super().__init__()
        if trans_type in ['linear', 'mlp']:
            self.model = TransitionSimple(z_dim, action_dim, trans_type=trans_type)
        elif 'reparam_w' in trans_type:
            self.model = TransitionParam(z_dim, action_dim, hidden_sizes=[64, 64],
                                         orthogonalize_mode=trans_type)
        else:
            raise Exception('Invalid trans_type:', trans_type)

    def forward(self, z, a):
        return self.model(z, a)


class TransitionSimple(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, trans_type='linear'):
        super().__init__()
        self.trans_type = trans_type
        self.z_dim = z_dim

        if self.trans_type == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        elif self.trans_type == 'mlp':
            hidden_size = 64
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, z_dim)
            )
        else:
            raise Exception('Invalid trans_type', trans_type)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        x = self.model(x)
        return x


class TransitionParam(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, hidden_sizes=[], orthogonalize_mode='reparam_w'):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.orthogonalize_mode = orthogonalize_mode

        if orthogonalize_mode == 'reparam_w_ortho_cont':
            self.model = MLP(z_dim + action_dim, z_dim * (z_dim - 1), hidden_sizes=hidden_sizes)
        else:
            self.model = MLP(z_dim + action_dim, z_dim * z_dim, hidden_sizes=hidden_sizes)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        if self.orthogonalize_mode == 'reparam_w':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim)  # b x z_dim x z_dim
        elif self.orthogonalize_mode == 'reparam_w_ortho_gs':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim)  # b x z_dim x z_dim
            Ws = orthogonalize_gs(Ws, self.z_dim)
        elif self.orthogonalize_mode == 'reparam_w_ortho_cont':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim - 1)  # b x z_dim x z_dim - 1
            Ws = orthogonalize_cont(Ws, self.z_dim)
        elif self.orthogonalize_mode == 'reparam_w_tanh':
            Ws = torch.tanh(self.model(x)).view(x.shape[0], self.z_dim, self.z_dim) / math.sqrt(self.z_dim)
        else:
            raise Exception('Invalid orthogonalize_mode:', self.orthogonalize_mode)
        return torch.bmm(Ws, z.unsqueeze(-1)).squeeze(-1) # b x z_dim

# Gram-Schmidt
def orthogonalize_gs(Ws, z_dim):
    Ws_new = Ws[:, :, [0]] / torch.norm(Ws[:, :, [0]], dim=1, keepdim=True)  # b x z_dim x 1
    for k in range(1, z_dim):
        v, us = Ws[:, :, [k]], Ws_new.permute(0, 2, 1)  # b x z_dim x 1, b x k x z_dim
        dot = torch.bmm(us, v)  # b x k x 1
        diff = (us * dot).sum(dim=1)  # b x z_dim
        u = Ws[:, :, k] - diff  # b x z_dim
        u = u / torch.norm(u, dim=1, keepdim=True)
        Ws_new = torch.cat((Ws_new, u.unsqueeze(-1)), dim=-1)
    return Ws_new


def orthogonalize_cont(Ws, z_dim):
    Ws_new = Ws[:, :, [0]] / torch.norm(Ws[:, :, [0]], dim=1, keepdim=True)  # b x z_dim x 1
    for k in range(1, z_dim - 1):
        v, us = Ws[:, :, [k]], Ws_new.permute(0, 2, 1)  # b x z_dim x 1, b x k x z_dim
        dot = torch.bmm(us, v)  # b x k x 1
        diff = (us * dot).sum(dim=1)  # b x z_dim
        u = Ws[:, :, k] - diff  # b x z_dim
        u = u / torch.norm(u, dim=1, keepdim=True)
        Ws_new = torch.cat((Ws_new, u.unsqueeze(-1)), dim=-1)

    # Ws_new is b x z_dim x z_dim - 1
    determinants = []
    for k in range(z_dim):
        tmp = torch.cat((Ws_new[:, :k], Ws_new[:, k+1:]), dim=1).permute(0, 2, 1).contiguous()
        tmp = tmp.cpu()
        det = torch.det(tmp)
        det = det.cuda()
        if k % 2 == 1:
            det = det * -1
        determinants.append(det)
    determinants = torch.stack(determinants, dim=-1).unsqueeze(-1) # b x z_dim x 1
    determinants = determinants / torch.norm(determinants, dim=1, keepdim=True)
    Ws_new = torch.cat((Ws_new, determinants), dim=-1)
    return Ws_new


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[]):
        super().__init__()
        model = []
        prev_h = input_size
        for h in hidden_sizes + [output_size]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop() # Pop last ReLU
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def quantize(x, n_bit):
    x = x * 0.5 + 0.5 # to [0, 1]
    x *= n_bit ** 2 - 1 # [0, 15] for n_bit = 4
    x = torch.floor(x + 1e-4) # [0, 15]
    return x
