import numpy as np
import torch
from torch.nn import functional as F
# import cv2
# from utils.gaussian_smoothing import GaussianSmoothing
from skimage import filters
import math

class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)



def fn_get_topk(attention_map, K=1):
    H, W = attention_map.size()
    attention_map_detach = attention_map.detach().view(H * W)
    topk_value, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []

    for index in topk_index:
        index = index.cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
    return topk_coord_list, topk_value


def fn_smoothing_func(attention_map):
    smoothing = GaussianSmoothing().to(attention_map.device)
    attention_map = F.pad(attention_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    attention_map = smoothing(attention_map).squeeze(0).squeeze(0)
    return attention_map

def fn_show_mask_attention(
    self_attention_maps,
    attention_res=16,
    smooth_attentions=True,
    mask=None
):

    self_attention_map_per_token_list = []
    H, W = self_attention_maps.shape[:2]
    
    if mask is not None:
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        mask = mask.squeeze().bool()
    else:
        mask = torch.ones(H, W, dtype=torch.bool, device=self_attention_maps.device)
    
    for coord_x in range(H):
        for coord_y in range(W):
            if mask[coord_x, coord_y]:
                self_attention_map_per_token = self_attention_maps[coord_x, coord_y]
                self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
                
                if smooth_attentions:
                    self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)

                norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
                    (self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
                
                self_attention_map_per_token_list.append(norm_self_attention_map_per_token)

    if len(self_attention_map_per_token_list) > 0:
        self_attention_map_numpy = torch.stack(self_attention_map_per_token_list, dim=0).cpu().detach().numpy()
    else:
        self_attention_map_numpy = np.array([])


    if len(self_attention_map_per_token_list) > 0:
        combined_attention = np.sum(self_attention_map_numpy, axis=0)
        combined_attention = (combined_attention - combined_attention.min()) / (combined_attention.max() - combined_attention.min() + 1e-6)
        self_attention_map_numpy = combined_attention[np.newaxis, ...]

    if len(self_attention_map_numpy) > 0:
        mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                                   size=(attention_res, attention_res), 
                                   mode='nearest').squeeze().cpu().numpy()
        
        attention_map = self_attention_map_numpy[-1]

        mask_score = np.mean(attention_map[mask_resized > 0])
        non_mask_score = np.mean(attention_map[mask_resized == 0])

        return self_attention_map_numpy, mask_score, non_mask_score
    return np.array([]), 0.0, 0.0





def fn_show_attention(
    cross_attention_maps,
    self_attention_maps,
    indices,
    K=1,
    attention_res=16,
    smooth_attentions=True,
):


    cross_attention_map_list, self_attention_map_list = [], []

    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

    indices = [index - 1 for index in indices]


    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
        cross_attention_map_list.append(cross_attention_map_per_token)

    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        topk_coord_list, topk_value = fn_get_topk(cross_attention_map_per_token, K=K)

        self_attention_map_per_token_list = []
        for coord_x, coord_y in topk_coord_list:

            self_attention_map_per_token = self_attention_maps[coord_x, coord_y]
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
            self_attention_map_per_token_list.append(self_attention_map_per_token)

        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / len(self_attention_map_per_token_list)
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
            (self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        
        self_attention_map_list.append(norm_self_attention_map_per_token)

    cross_attention_map_numpy       = torch.cat(cross_attention_map_list, dim=0).cpu().detach().numpy()
    self_attention_map_numpy        = torch.cat(self_attention_map_list, dim=0).cpu().detach().numpy()

    return cross_attention_map_numpy, self_attention_map_numpy




def fn_get_otsu_mask(x):

    x_numpy = x
    x_numpy = x_numpy.cpu().detach().numpy()
    x_numpy = x_numpy * 255
    x_numpy = x_numpy.astype(np.uint16)

    opencv_threshold = filters.threshold_otsu(x_numpy)
    opencv_threshold = opencv_threshold * 1. / 255.

    otsu_mask = torch.where(
        x < opencv_threshold,
        torch.tensor(0, dtype=x.dtype, device=x.device),
        torch.tensor(1, dtype=x.dtype, device=x.device))
    
    return otsu_mask


def fn_clean_mask(otsu_mask, x, y):
    
    H, W = otsu_mask.size()
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def dfs(cur_x, cur_y):
        if cur_x >= 0 and cur_x < H and cur_y >= 0 and cur_y < W and otsu_mask[cur_x, cur_y] == 1:
            otsu_mask[cur_x, cur_y] = 2
            for delta_x, delta_y in direction:
                dfs(cur_x + delta_x, cur_y + delta_y)
    
    dfs(x, y)
    ret_otsu_mask = torch.where(
        otsu_mask < 2,
        torch.tensor(0, dtype=otsu_mask.dtype, device=otsu_mask.device),
        torch.tensor(1, dtype=otsu_mask.dtype, device=otsu_mask.device))

    return ret_otsu_mask