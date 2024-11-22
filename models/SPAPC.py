from timm.models.layers import DropPath
from utils.util import *
from extensions.chamfer_dist import ChamferDistanceL1
from .build import MODELS
from utils.misc import generate_image

def fps(pc, num, if_idx=False):
    fps_idx = furthest_point_sample(pc, num)
    sub_pc = gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

    if if_idx:
        return fps_idx.type(torch.int64)

    return sub_pc

def knn(x_q, x_k, num_sample):
    _, nq, _ = x_q.shape
    _, nk, _ = x_k.shape
    rel_dist = x_q.unsqueeze(2).repeat(1, 1, nk, 1) - x_k.unsqueeze(1).repeat(1, nq, 1, 1)
    rel_dist = torch.norm(input=rel_dist, dim=-1, p=2)
    idx = rel_dist.topk(k=num_sample, dim=-1)[1]

    return idx

def get_graph_region(x_q, x_k, n_sample):
    b, n, c = x_q.shape
    device = torch.device(x_q.device)

    idx = knn(x_q=x_q, x_k=x_k, num_sample=n_sample)  # (b, n, k)
    idx_base = torch.arange(0, b, device=device).view(-1, 1, 1) * n
    idx = idx + idx_base
    idx = idx.view(-1)
    center = x_q.contiguous()
    neighbor = x_k.contiguous()
    neighbor = neighbor.view(b * n, -1)[idx, :]  # (b, n, c)  -> (b*n, c)
    neighbor = neighbor.view(b, n, n_sample, c)  # b * n * k + range(0, b*n)
    center = center.unsqueeze(2).repeat(1, 1, n_sample, 1)
    feature = torch.cat((neighbor - center, center), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (b, c*2, n, k)

class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class EdgeFeature(nn.Module):
    def __init__(self, inchannels, outchannels, mode='m'):
        super(EdgeFeature, self).__init__()
        self.mode = mode
        self.conv = nn.Sequential(nn.Conv2d(inchannels * 2, outchannels, kernel_size=(1, 1), bias=False),
                                  nn.BatchNorm2d(outchannels),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x_q, x_k=None, n_sample=8):
        x_k = x_q if x_k==None else x_k
        feature = get_graph_region(x_q, x_k, n_sample)  # (b, c, n) -> (b, c*2, n, k)
        feature = self.conv(feature)  # (b, c*2, n, k) -> (b, c, n, k)

        if self.mode == 'm':
            feature = feature.max(dim=-1, keepdim=False)[0]  # (b, c, n, k) -> (b, c, n)
        else:
            feature = torch.mean(feature, dim=-1)  # (b, c, n, k) -> (b, c, n)

        return feature

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.position_mlp = nn.Sequential
        self.q_conv = nn.Conv1d(channels, channels // 4, (1,), bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, (1,), bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, (1,))
        self.trans_conv = nn.Conv1d(channels, channels, (1,))
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Location_Correct(nn.Module):
    def __init__(self, dim=256, num_heads=8, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.reduce_map = nn.Sequential(
            nn.Linear(dim + 3, dim, 1),
            nn.ReLU(inplace=True)
        )

        self.coarse_pred = nn.Sequential(
            nn.Linear(dim, dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 3, 1)
        )


    def forward(self, coarse, ms_feature, sub_feature):
        ms_feature = self.reduce_map(torch.cat([ms_feature, coarse], dim=-1))
        norm_q = self.norm1(ms_feature)
        q_1 = self.self_attn(norm_q)

        ms_feature = ms_feature + self.drop_path(q_1)

        norm_q = self.norm_q(ms_feature)
        norm_v = self.norm_v(sub_feature)
        q_2 = self.attn(norm_q, norm_v)

        ms_feature = ms_feature + self.drop_path(q_2)
        ms_feature = ms_feature + self.drop_path(self.mlp(self.norm2(ms_feature)))
        coarse = self.coarse_pred(ms_feature)

        return  coarse, ms_feature.transpose(1, 2)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False)
        self.transformer_1 = vTransformer(128, dim=64, n_knn=16)
        self.sa_module_2 = PointNet_SA_Module_KNN(384, 16, 128, [128, 256], group_all=False, if_bn=False)
        self.transformer_2 = vTransformer(256, dim=64, n_knn=16)

        self.edge_conv_1 = EdgeFeature(3, 64)
        self.edge_conv_2 = EdgeFeature(64, 128)
        self.edge_conv_3 = EdgeFeature(128, 256)
        self.pt_posemb = nn.Sequential(
            nn.Conv1d(3, 128, (1,)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 256, (1,))
        )

    def forward(self, partial, junction_index):
        pt = partial  # (b, 3, n)

        sub_xyz, sub_feature = self.sa_module_1(pt, pt)
        sub_feature = self.transformer_1(sub_feature, sub_xyz)
        sub_xyz, sub_feature = self.sa_module_2(sub_xyz, sub_feature)
        sub_feature = self.transformer_2(sub_feature, sub_xyz)
        bd = torch.gather(partial, dim=-1, index=junction_index.unsqueeze(1).repeat(1, 3, 1))  # (b, 3, m)

        rel_feature = self.edge_conv_1(x_q=bd.permute(0, 2, 1), x_k=sub_xyz.permute(0, 2, 1))
        rel_feature = self.edge_conv_2(rel_feature.permute(0, 2, 1))
        rel_feature = self.edge_conv_3(rel_feature.permute(0, 2, 1))

        return sub_xyz.permute(0, 2, 1), bd.permute(0, 2, 1), sub_feature, rel_feature

class Decoder(nn.Module):
    def __init__(self, latent_dim,  num_coarse=384, num_dense=6144):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_dense = num_dense
        self.num_coarse = num_coarse
        self.fold_step = int(pow(self.num_dense // self.num_coarse, 0.5) + 0.5)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(latent_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.increase_dim2 = nn.Sequential(
            nn.Conv1d(latent_dim+3, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.bias_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_coarse)
        )

        self.reduce_map2 = nn.Sequential(
            nn.Linear(1024+latent_dim+3, latent_dim),
            nn.ReLU(inplace=True)
        )


        self.coarse_corrector = nn.ModuleList([Location_Correct() for i in range(3)])

        self.foldingnet = Fold(self.latent_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point

    def forward(self, sub_xyz, bd, sub_feature, rel_feature):
        b, c, n = sub_feature.shape
        coarse_bias = self.increase_dim(rel_feature)
        coarse_bias = torch.max(coarse_bias, dim=-1)[0]
        coarse = bd + self.bias_pred(coarse_bias).reshape(b, -1, 3)  #  B M C(3)
        ms_feature = sub_feature

        for corrector in self.coarse_corrector:
            coarse, ms_feature = corrector(coarse, ms_feature.transpose(1, 2), sub_feature.transpose(1,2))

        global_feature = self.increase_dim2(torch.cat([coarse.transpose(1, 2), ms_feature], dim=1)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, n, -1),
            ms_feature.transpose(1, 2),
            coarse], dim=-1)  # B M 1027 + C

        rebuild_feature = self.reduce_map2(rebuild_feature.reshape(b*n, -1)) # BM C
        #
        # # NOTE: foldingNet
        relative_xyz = self.foldingnet(rebuild_feature).reshape(b, n, 3, -1)    # B M 3 S
        dense = (relative_xyz + coarse.unsqueeze(-1)).transpose(2,3).reshape(b, -1, 3)  # B N 3

        return coarse, dense

@MODELS.register_module()
class SPAPC(nn.Module):
    def __init__(self, config):
        super(SPAPC, self).__init__()
        self.num_dense = config.num_dense
        self.num_coarse = config.num_coarse
        self.encoder = Encoder(config.latent_dim)
        self.decoder = Decoder(config.latent_dim, self.num_coarse, self.num_dense)
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, partial, junction_index):
        sub_xyz, bd, sub_feature, bd_feature = self.encoder(partial.permute(0, 2, 1).contiguous(),
                                                                        junction_index)
        coarse, dense = self.decoder(sub_xyz, bd, sub_feature, bd_feature)
        return (coarse, dense)



