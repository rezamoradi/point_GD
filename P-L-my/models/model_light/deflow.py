import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import List
from enum import Enum
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.ops import knn_gather
from modules.utils.distribution import GaussianDistribution
from models.layers import ActNorm


from models.model_light.layer import noiseEdgeConv, EdgeConv, FeatMergeUnit, FullyConnectedLayer, PreConv
from metric.loss import MaskLoss, ConsistencyLoss
import models.layers.base as base_layers
import models.layers as layers
class Disentanglement(Enum):
    FBM = 1
    LBM = 2
    LCC = 3

import open3d as o3d
import numpy as np


ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base_layers.Swish(),
    'LeakyLSwish': lambda b: base_layers.LeakyLSwish(),
    'CLipSwish': lambda b: base_layers.CLipSwish(),
    'ALCLipSiLU': lambda b: base_layers.ALCLipSiLU(),
    'pila': lambda b: base_layers.Pila(),
    'CPila': lambda b: base_layers.CPila(),
    'lcube': lambda b: base_layers.LipschitzCube(),
    'identity': lambda b: base_layers.Identity(),
    'relu': lambda b: base_layers.MyReLU(inplace=b),
    'CReLU': lambda b: base_layers.CReLU(),
}


parser = argparse.ArgumentParser(add_help=False)
args = parser.parse_args()

process_point_cloud_diffusion = MyDiffusion(args)
print(args.diffusion_dir)
        

def save_point_cloud_as_ply(point_cloud, file_path):
    """
    Save a point cloud as a .ply file.

    Parameters:
    - point_cloud: numpy array or torch tensor of shape (N, 3)
    - file_path: str, the path where the .ply file will be saved
    """
    # Convert the point cloud to a numpy array if it's a torch tensor
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Save the point cloud as a .ply file
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Point cloud saved to {file_path}")


@torch.enable_grad()
def cloudfixer(args, model, x, mask, ind, verbose=False):
    ######################## Scheduler ########################
    def get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps,
        last_epoch=-1,
        end_factor=0,
    ):

        """
        Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period.
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return end_factor + max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    ######################## End Scheduler ########################
    _, knn_dist_square_mean = knn(
        x.transpose(2, 1),
        k=args.knn,
        mask=(mask.squeeze(-1).bool()),
        ind=ind,
        return_dist=True,
    )
    
    knn_dist_square_mean = knn_dist_square_mean[torch.arange(x.size(0))[:, None], ind]
    weight = 1 / knn_dist_square_mean.pow(args.pow)
    if not args.weighted_reg:
        weight = torch.ones_like(weight)
    weight = weight / weight.sum(dim=-1, keepdim=True)  # normalize
    weight = weight * mask.squeeze(-1)
    node_mask = x.new_ones(x.shape[:2]).to(x.device).unsqueeze(-1)
    delta = torch.nn.Parameter(torch.zeros_like(x))
    rotation = torch.nn.Parameter(x.new_zeros((x.size(0), 6)))
    rotation_base = x.new_zeros((x.size(0), 6))
    rotation_base[:, 0] = 1
    rotation_base[:, 4] = 1
    delta.requires_grad_(True)
    rotation.requires_grad_(True)

    optim = torch.optim.Adamax(
        [
            {"params": [delta], "lr": args.input_lr},
            {"params": [rotation], "lr": args.rotation},
        ],
        lr=args.input_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    scheduler = get_linear_schedule_with_warmup(
        optim,
        int(args.n_update * args.warmup),
        args.n_update,
        last_epoch=-1,
        end_factor=args.optim_end_factor,
    )

    loss_type = "cdl1"
    if loss_type == "cdl1":
        loss_func = ChamferDistanceL1().cuda()
    elif loss_type == 'cdl2':
        loss_func = ChamferDistanceL2().cuda()


    iterator = tqdm(range(args.n_update)) if verbose else range(args.n_update)
    for iter in iterator:
        optim.zero_grad()
        t = args.t_min + args.t_len * torch.rand(x.shape[0], 1).to(x.device)
        t = (t * args.diffusion_steps).long().float() / args.diffusion_steps
        gamma_t = model.module.inflate_batch_array(model.module.gamma(t), x)
        alpha_t = model.module.alpha(gamma_t, x)  # batch_size x 1 x 1
        sigma_t = model.module.sigma(gamma_t, x)

        eps = torch.randn_like(x)
        x_trans = x + delta
        rot = compute_rotation_matrix_from_ortho6d(rotation + rotation_base)
        x_trans = x_trans @ rot
        z_loss = 0
        with torch.no_grad():
            x_trans_t = x_trans * alpha_t + sigma_t * eps
            _, x_trans_est = model(
                x_trans_t,
                phi=True,
                return_x0_est=True,
                t=t,
                node_mask=node_mask,
            )
        # dist1, dist2, _, _ = chamfer_dist_fn(x_trans, x_trans_est)
        # matching = dist1.mean() + dist2.mean()

        matching = loss_func(x_trans, x_trans_est)
        L2_norm = (delta.pow(2) * weight[:, :, None]).sum(dim=1).mean()
        norm = L2_norm * (
            args.lam_h * (1 - iter / args.n_update) + args.lam_l * iter / args.n_update
        )
        loss = matching + norm + z_loss
        loss.backward()
        optim.step()
        scheduler.step()
        if verbose and (iter) % 10 == 0:
            print("LR", scheduler.get_last_lr())
            print("rotation", (rotation_base + rotation).abs().mean(dim=0))
            print("delta", (delta).abs().mean().item())  # norm(2, dim=-1).mean()
            print(
                delta[mask.expand_as(delta) == 1].abs().mean().item(),
                delta[mask.expand_as(delta) == 0].abs().mean().item(),
            )
    rot = compute_rotation_matrix_from_ortho6d(rotation + rotation_base)
    x_trans = x + delta
    x_trans = x_trans @ rot
    if verbose:
        print("LR", scheduler.get_last_lr())
        print("rotation", (rotation_base + rotation).abs().mean(dim=0))
        print("delta", (delta).norm(2, dim=-1).mean())

    return x_trans


class MyDiffusion(nn.Module):
    def __init__(self, args):
        super(MyDiffusion, self).__init__()

        args.vote = 1
        args.n_update = 80
        args.pow = 1
        args.knn = 5
        args.input_lr = 0.2 #1e-2
        args.rotation = 0.02
        args.weight_decay = 0
        args.weighted_reg = True
        args.warmup = 0.2
        args.optim_end_factor= 0.05
        args.beta1 = 0.9
        args.beta2 = 0.999


        args.diffusion_model = 'transformer'
        args.probabilistic_model = 'diffusion'
        
        args.diffusion_steps = 500
        args.diffusion_noise_schedule = "polynomial_2"
        args.diffusion_noise_precision = 1e-5
        args.diffusion_loss_type = "l2"
        args.diffusion_dir = args.model_path+"diffusion.npy"
        # args.diffusion_dir = args.model_path+"diffusion_release.npy"
        # args.diffusion_dir = args.model_path+"diffusion_modelnet.npy"
        # args.diffusion_dir = args.model_path+"diffusion_scannet.npy"
        # args.diffusion_dir = args.model_path+"diffusion_shapecore.npy"
        # args.diffusion_dir = args.model_path+"diffusion_shapenet.npy"

        args.lam_h = 10
        args.lam_l = 1

        args.t_min = 0.02
        args.t_len = 0.1

        
        self.args = args
        self.model = get_model(args, device)
        print('self.model ',self.model)
        self.model.load_state_dict(torch.load(args.diffusion_dir, map_location="cpu"))
        print('self.model ',self.model)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device).eval()

    def forward(self, x, mask, ind, update1=30):        

        self.args.n_update = update1

        x_list = [
            cloudfixer(self.args, self.model, x, mask, ind).detach().clone()
            for v in range(self.args.vote)
        ]
        x = x_list[0]

        # x = torch.asarray(inv_rotate_pc(x.cpu().detach().numpy())).to(args.device)
        # save_point_cloud_as_ply(x[0].cpu().detach().numpy(), f'/content/b1.ply')

        x = inv_rotate_pc(x)
        # save_point_cloud_as_ply(x[0].cpu().detach().numpy(), f'/content/b2.ply')

        x = pc_normalize(x)
        # save_point_cloud_as_ply(x[0].cpu().detach().numpy(), f'/content/b3.ply')
        
        return x 











class DenoiseFlow(nn.Module):

    def __init__(
        self,
        disentangle: Disentanglement,
        pc_channel=3,

        # aug_channel=8,
        # n_aug = 4,
        aug_channel=32,

        n_injector = 7,
        num_neighbors = 32,

        cut_channel=6,
        nflow_module=8,
        coeff=0.9,
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        activation_fn='elu',
        n_exact_terms=0,
        neumann_grad=True,
        grad_in_forward=False,

        nhidden=2,
        idim=64,
        densenet=False,
        densenet_depth=3,
        densenet_growth=32,
        learnable_concat=False,
        lip_coeff=0.98,

    ):
        super(DenoiseFlow, self).__init__()

        self.disentangle = disentangle
        self.pc_channel = pc_channel

        self.aug_channel = aug_channel
        # self.n_aug = n_aug
        self.n_injector = n_injector
        self.num_neighbors = num_neighbors
        self.cut_channel = cut_channel
        self.nflow_module = nflow_module
        self.coeff = coeff
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.activation_fn = activation_fn
        self.n_exact_terms = n_exact_terms
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.nhidden = nhidden
        self.idim = idim
        self.densenet = densenet
        self.densenet_depth = densenet_depth
        self.densenet_growth = densenet_growth
        self.learnable_concat = learnable_concat
        self.lip_coeff = lip_coeff

        self.dist = GaussianDistribution()
        self.noise_params = noiseEdgeConv(pc_channel, hidden_channel=32, out_channel=aug_channel)
        self.feat_Conv = nn.ModuleList()
        self.AdaptConv = nn.ModuleList()
        self.PreConv = PreConv(in_channel=self.pc_channel, out_channel=16)
        in_channelE = [16, 48, 80, 112, 144, 176, 208, 240, 96, 120, 144, 168]
        in_channelA = [48, 80, 112, 144, 176, 208, 240, 96, 120, 144, 168, 96]
        hidden_channel = 64
        out_channel = [32, 32, 32, 32, 32, 32, 32, 96, 24, 24, 24, 96]
        for i in range(self.n_injector):
            concat = False if i == 7 or i == 11 else True
            self.feat_Conv.append(
                EdgeConv(in_channelE[i], hidden_channel=hidden_channel, out_channel=out_channel[i], concat=concat))
            self.AdaptConv.append(FeatMergeUnit(in_channel=in_channelA[i], hidden_channel=hidden_channel,
                                                out_channel=self.pc_channel + self.aug_channel))


        flow_assemblies = []
        for i in range(self.nflow_module):
            flow = FlowAssembly(
                id=i,
                nflow_module=self.nflow_module,
                channel=self.pc_channel + self.aug_channel,
                coeff=self.coeff,
                n_lipschitz_iters=self.n_lipschitz_iters,
                sn_atol=self.sn_atol,
                sn_rtol=self.sn_rtol,
                n_power_series=self.n_power_series,
                n_dist=self.n_dist,
                n_samples=self.n_samples,
                activation_fn=self.activation_fn,
                n_exact_terms=self.n_exact_terms,
                neumann_grad=self.neumann_grad,
                grad_in_forward=self.grad_in_forward,
                nhidden=self.nhidden,
                idim=self.idim,
                densenet=self.densenet,
                densenet_depth=self.densenet_depth,
                densenet_growth=self.densenet_growth,
                learnable_concat=self.learnable_concat,
                lip_coeff=self.lip_coeff,
            )
            flow_assemblies.append(flow)
        self.flow_assemblies = nn.ModuleList(flow_assemblies)
        # -----------------------------------------------
        # Disentangle method
        if self.disentangle == Disentanglement.FBM:  # Fix binary mask
            self.channel_mask = nn.Parameter(torch.ones((1, 1, self.pc_channel + self.aug_channel)),
                                             requires_grad=False)
            self.channel_mask[:, :, -self.cut_channel:] = 0.0

        if self.disentangle == Disentanglement.LBM:  # Learnable binary mask
            self.mloss = MaskLoss()
            theta = torch.rand((1, 1, self.pc_channel + self.aug_channel))
            self.theta = nn.Parameter(theta, requires_grad=True)

        if self.disentangle == Disentanglement.LCC:  # Latent Code Consistency
            self.closs = ConsistencyLoss()
            # Random initialization
            w_init = np.random.randn(self.pc_channel + self.aug_channel, self.pc_channel + self.aug_channel)
            w_init = np.linalg.qr(w_init)[0].astype(np.float32)
            self.channel_mask = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)

    def f(self, x: Tensor, inj_f: List[Tensor]):
        B, N, _ = x.shape
        log_det_J = torch.zeros((B,), device=x.device)
        for i in range(self.nflow_module):
            if i < self.n_injector:
                x = x + inj_f[i]


            x = self.flow_assemblies[i].forward(x)

        return x, log_det_J

    def g(self, z: Tensor, inj_f: List[Tensor]):
        for i in reversed(range(self.nflow_module)):
            z = self.flow_assemblies[i].inverse(z)
            if i < self.n_injector:
                z = z - inj_f[i]

        return z

    def log_prob(self, xyz: Tensor, inj_f: List[Tensor]):

        aug_feat = self.unit_coupling(xyz)
        x = torch.cat([xyz, aug_feat], dim=-1)  # [B, N, 3 + C]
        z, flow_ldj = self.f(x, inj_f)
        logp = 0
        return z, logp



    def sample(self, z: Tensor, inj_f: List[Tensor]):
        full_x = self.g(z, inj_f)
        clean_x = full_x[..., :self.pc_channel]  # [B, N, 3]
        return clean_x

    def forward(self, x: Tensor, y: Tensor = None):
        p = x
        inj_f = self.feat_extract(p)
        z, ldj = self.log_prob(x, inj_f)

        loss_denoise = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        if self.disentangle == Disentanglement.FBM:  # Fix channel mask
            z[:, :, -self.cut_channel:] = 0
            predict_z = z
            # or
            # predict_z = z * self.channel_mask

        if self.disentangle == Disentanglement.LBM:
            # Learnable binary mask
            mask = torch.max(torch.zeros_like(self.theta), 1.0 - (-self.theta).exp())
            mask = 1.0 - (-self.theta).exp()
            predict_z = z * mask
            loss_denoise = self.mloss(mask)

        if self.disentangle == Disentanglement.LCC:  # Latent Code Consistency
            clean_z, _ = self.log_prob(y, inj_f) if y is not None else (None, None)

            # Identity initialization
            predict_z = torch.einsum('ij,bnj->bni', self.channel_mask, z)
            # Random initialization
            # predict_z = z * self.channel_mask.expand_as(z)
            # loss_denoise = self.closs(predict_z, clean_z) if y is not None else None
         
         
        save_point_cloud_as_ply(predict_z[0].cpu().detach().numpy(), f'/content/b1.ply')
        print('predict_z ',predict_z.shape)
        
        # NUM_POINTS = 1024
        # pointcloud = predict_z[].clone        
        # N = len(pointcloud[0,:,0])
        # # mask = torch.ones((max(NUM_POINTS, N), 1)).astype(pointcloud.dtype)
        # mask = torch.ones((max(NUM_POINTS, N), 1))
        # mask[N:] = 0
        
        # ind = np.arange(len(pointcloud[0,:,0]))
        # while len(pointcloud) < NUM_POINTS:
            # chosen = np.arange(N)
            # np.random.shuffle(chosen)
            # chosen = chosen[: NUM_POINTS - len(pointcloud[0,:,0])]
            # pointcloud = np.concatenate(
                # (pointcloud, pointcloud[chosen]), axis=0
            # )
            # ind = np.concatenate((ind, chosen), axis=0)

        # predict_z = process_point_cloud_diffusion(pointcloud, mask, ind)
        # save_point_cloud_as_ply(predict_z[0].cpu().detach().numpy(), f'/content/b2.ply')
        # print('predict_z ',predict_z.shape)
        



        predict_x = self.sample(predict_z, inj_f)
        print('predict_x ',predict_x.shape)
        save_point_cloud_as_ply(predict_x[0].cpu().detach().numpy(), f'/content/b3.ply')
        
        return predict_x, ldj, loss_denoise

    def unit_coupling(self, xyz):
        B, N, _ = xyz.shape
        _, knn_idx, _ = knn_points(xyz, xyz, K=self.num_neighbors, return_nn=False, return_sorted=False)
        f = self.noise_params(xyz, knn_idx)
        return f

    def feat_extract(self, xyz: Tensor):
        cs = []
        _, knn_idx, _ = knn_points(xyz, xyz, K=self.num_neighbors, return_nn=False, return_sorted=False)
        f = self.PreConv(xyz, knn_idx)
        for i in range(self.n_injector):
            f = self.feat_Conv[i](f, knn_idx)
            inj_f = self.AdaptConv[i](f)
            cs.append(inj_f)
        return cs

    def nll_loss(self, pts_shape, sldj):
        # ll = sldj - np.log(self.k) * torch.prod(pts_shape[1:])
        # ll = torch.nan_to_num(sldj, nan=1e3)
        ll = sldj

        nll = -torch.mean(ll)

        return nll

    def denoise(self, noisy_pc: Tensor):
        clean_pc, _, _ = self(noisy_pc)
        return clean_pc

    def init_as_trained_state(self):
        """Set the network to initialized state, needed for evaluation(significant performance impact)"""
        for i in range(self.nflow_module):
            self.flow_assemblies[i].chain[1].is_inited = True
            self.flow_assemblies[i].chain[3].is_inited = True


class FlowAssembly(layers.SequentialFlow):

    def __init__(
            self,
            id,
            nflow_module,
            channel,
            coeff=0.9,
            n_lipschitz_iters=None,
            sn_atol=None,
            sn_rtol=None,
            n_power_series=5,
            n_dist='geometric',
            n_samples=1,
            activation_fn='elu',
            n_exact_terms=0,
            neumann_grad=True,
            grad_in_forward=False,
            nhidden=2,
            idim=64,
            densenet=False,
            densenet_depth=3,
            densenet_growth=32,
            learnable_concat=False,
            lip_coeff=0.98,

    ):
        chain = []

        def _quadratic_layer(channel):
            return layers.InvertibleLinear(channel)

        def _actnorm(channel):
            return ActNorm(channel)

        def _lipschitz_layer():
            return base_layers.get_linear

        def _iMonotoneBlock(preact=False):
            return layers.iMonotoneBlock(
                FCNet(
                    preact=preact,
                    channel=channel,
                    lipschitz_layer=_lipschitz_layer(),
                    coeff=coeff,
                    n_iterations=n_lipschitz_iters,
                    activation_fn=activation_fn,
                    sn_atol=sn_atol,
                    sn_rtol=sn_rtol,
                    nhidden=nhidden,
                    idim=idim,
                    densenet=densenet,
                    densenet_depth=densenet_depth,
                    densenet_growth=densenet_growth,
                    learnable_concat=learnable_concat,
                    lip_coeff=lip_coeff,
                ),
                n_power_series=n_power_series,
                n_dist=n_dist,
                n_samples=n_samples,
                n_exact_terms=n_exact_terms,
                neumann_grad=neumann_grad,
                grad_in_forward=grad_in_forward,
            )
        chain.append(_iMonotoneBlock())
        chain.append(_actnorm(channel))
        chain.append(_iMonotoneBlock(preact=True))
        chain.append(_actnorm(channel))


        super(FlowAssembly, self).__init__(chain)

class FCNet(nn.Module):

    def __init__(
        self, preact, channel, lipschitz_layer, coeff, n_iterations, activation_fn, sn_atol, sn_rtol,
        nhidden=2, idim=64, densenet=False, densenet_depth=3, densenet_growth=32, learnable_concat=False, lip_coeff=0.98
    ):
        super(FCNet, self).__init__()
        nnet = []
        last_dim = channel
        if not densenet:
            if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU']:
                idim_out = idim // 2
                last_dim_in = last_dim * 2
            else:
                idim_out = idim
                last_dim_in = last_dim
            if(preact):
                nnet.append(ACT_FNS[activation_fn](False))
            for i in range(nhidden):
                nnet.append(
                    lipschitz_layer(
                        last_dim_in, idim_out, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                        atol=sn_atol, rtol=sn_rtol
                    )
                )
                nnet.append(ACT_FNS[activation_fn](True))
                last_dim_in = idim_out * 2 if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU'] else idim_out
            nnet.append(
                lipschitz_layer(
                    last_dim_in, last_dim, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )
        else:
            first_channels = 64

            nnet.append(
                lipschitz_layer(
                    channel, first_channels, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )

            total_in_channels = first_channels

            for i in range(densenet_depth):
                part_net = []

                # Change growth size for CLipSwish:
                if activation_fn in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU']:
                    output_channels = densenet_growth // 2
                else:
                    output_channels = densenet_growth

                part_net.append(
                    lipschitz_layer(
                        total_in_channels, output_channels, coeff=coeff, n_iterations=n_iterations, domain=2,
                        codomain=2, atol=sn_atol, rtol=sn_rtol
                    )
                )

                part_net.append(ACT_FNS[activation_fn](True))

                nnet.append(
                    layers.LipschitzDenseLayer(layers.ExtendedSequential(*part_net),
                                               learnable_concat,
                                               lip_coeff
                                               )
                )

                total_in_channels += densenet_growth

            nnet.append(
                lipschitz_layer(
                    total_in_channels, last_dim, coeff=coeff, n_iterations=n_iterations, domain=2, codomain=2,
                    atol=sn_atol, rtol=sn_rtol
                )
            )

        self.nnet = layers.ExtendedSequential(*nnet)

    def forward(self, x):
        y = self.nnet(x)
        return y

    def build_clone(self):
        class FCNetClone(nn.Module):
            def __init__(self, nnet):
                super(FCNetClone, self).__init__()
                self.nnet = nnet

            def forward(self, x):
                y = self.nnet(x)
                return y

        return FCNetClone(self.nnet.build_clone())

    def build_jvp_net(self, x):
        class FCNetJVP(nn.Module):
            def __init__(self, nnet):
                super(FCNetJVP, self).__init__()
                self.nnet = nnet

            def forward(self, v):
                jv = self.nnet(v)
                return jv

        nnet, y = self.nnet.build_jvp_net(x)
        return FCNetJVP(nnet), y


# -----------------------------------------------------------------------------------------
class DenoiseFlowMLP(DenoiseFlow):

    def __init__(self, disentangle: Disentanglement, pc_channel=3):
        super(DenoiseFlowMLP, self).__init__(disentangle, pc_channel)

        full_channel = self.in_channel + self.aug_channel * self.n_aug

        self.sample_mlp = nn.Sequential(
            FullyConnectedLayer(full_channel, full_channel * 2, activation='relu'),
            FullyConnectedLayer(full_channel * 2, full_channel * 4, activation='relu'),
            FullyConnectedLayer(full_channel * 4, full_channel * 4, activation='relu'),
            FullyConnectedLayer(full_channel * 4, full_channel * 2, activation='relu'),
            FullyConnectedLayer(full_channel * 2, full_channel * 2, activation='relu'),
            FullyConnectedLayer(full_channel * 2, full_channel, activation='relu'),
            FullyConnectedLayer(full_channel, full_channel, activation='relu'),
            FullyConnectedLayer(full_channel, full_channel // 2, activation='relu'),
            FullyConnectedLayer(full_channel // 2, 3, activation=None))

    def sample(self, z: Tensor, Edgef: List[Tensor]):
        return self.sample_mlp(z)
# -----------------------------------------------------------------------------------------
