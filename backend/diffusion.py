import torch
from ml.ldm import LDM

diff_model = torch.load('models/ldm-basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)-0005-100-1400.pkl')

def sample_glyphs(feature_im : torch.Tensor = None, mask : torch.Tensor = None):
    latent_shape = (1, 26, 2048)
    return diff_model.sample(latent_shape)