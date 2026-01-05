import numpy as np
import torch
import sys
import os
bart_path = '/data0/tianyu/bart'
if os.path.join(bart_path, 'python') not in sys.path:
    sys.path.insert(0, os.path.join(bart_path, 'python'))

os.environ['TOOLBOX_PATH'] = bart_path
from bart import bart

def ifft2_np(x: np.ndarray) -> np.ndarray:
    """Centered 2D IFFT (ortho)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

def fft2_np(x: np.ndarray) -> np.ndarray:
    """Centered 2D FFT (ortho)."""
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

def to_tensor(data: np.ndarray) -> torch.Tensor:
    if np.iscomplexobj(data):        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def kspace_to_target(x: np.ndarray) -> np.ndarray:
    """Generate RSS reconstruction target from k-space data.

    Args:
        x: K-space data

    Returns:
        RSS reconstruction target
    """
    return np.sqrt(np.sum(np.square(np.abs(ifft2_np(x))), axis=-3)).astype(np.float32)
def ifft2c_tensor(x):
    """ifft2c with tensor as input/output, used for k-space to image.

    Shape:
        Input: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
        Output: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
    """
    x = torch.complex(x[...,0], x[...,1]).numpy()
    x = ifft2_np(x)
    x = torch.from_numpy(x)
    x = torch.stack((x.real, x.imag), dim=-1)
    return x

def fft2c_tensor(x):
    """fft2c with tensor as input/output, used for image to k-space.

    Shape:
        Input: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
        Output: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
    """
    x = torch.complex(x[...,0], x[...,1]).numpy()
    x = fft2_np(x)
    x = torch.from_numpy(x)
    x = torch.stack((x.real, x.imag), dim=-1)
    return x

def get_mvue(mctgt, S):
    mctgt = torch.view_as_complex(mctgt)
    S = torch.view_as_complex(S)
    numerator = torch.sum(mctgt * torch.conj(S), dim=1)
    denominator = torch.sum(S.abs().square(), dim=1)
    return numerator / (denominator + 1e-8)

def kspace_to_sensmaps_mvue(kspace: np.ndarray):
    """
    same as calculate_mvue_and_sens but use bart toolbox
    """
    mctgt = ifft2_np(kspace)
    mctgt = to_tensor(mctgt)
    mctgt = mctgt.unsqueeze(0)
    sens_maps = bart(1, "ecalib -m1 -c0 -r24 -W", kspace[None, ...].transpose(0, 2, 3, 1))
    print(sens_maps.shape)
    sens_maps = sens_maps.transpose(0, 3, 1, 2)
    S = to_tensor(sens_maps)
    mvue = get_mvue(mctgt, S)
    mvue = mvue.squeeze().cpu().numpy()
    sens_maps = sens_maps.squeeze(0)
    return mvue.astype(np.complex64), sens_maps.astype(np.complex64)