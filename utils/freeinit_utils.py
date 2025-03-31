import torch
import torch.fft as fft
import math


def freq_mix_2D(x, noise, LPF):
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    noise_freq = fft.fftn(noise, dim=(-2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-2, -1))

    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high

    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-2, -1)).real

    return x_mixed




def get_freq_filter(shape, device, filter_type, n, d_s, d_t=None):

    if filter_type == "gaussian":
        return gaussian_low_pass_filter_2D(shape=shape, d_s=d_s).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter_2D(shape=shape, d_s=d_s).to(device)
    elif filter_type == "box":
        return box_low_pass_filter_2D(shape=shape, d_s=d_s).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter_2D(shape=shape, n=n, d_s=d_s).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter_2D(shape, d_s=0.25):

    H, W = shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0:
        return mask
    for h in range(H):
        for w in range(W):
            d_square = ((2*h/H-1)**2 + (2*w/W-1)**2)
            mask[..., h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter_2D(shape, n=4, d_s=0.25):
    H, W = shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0:
        return mask
    for h in range(H):
        for w in range(W):
            d_square = ((2*h/H-1)**2 + (2*w/W-1)**2)
            mask[..., h,w] = 1 / (1 + (d_square / d_s**2)**n)

    return mask


def ideal_low_pass_filter_2D(shape, d_s=0.25):
    H, W = shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0:
        return mask
    for h in range(H):
        for w in range(W):
            d_square = ((2*h/H-1)**2 + (2*w/W-1)**2)
            mask[..., h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter_2D(shape, d_s=0.25):

    H, W = shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0:
        return mask
    threshold_s = round(int(H // 2) * d_s)
    crow, ccol = H // 2, W //2
    mask[..., crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask
