import torch
import os
print(os.getcwd(), __file__, os.path.dirname(__file__), os.listdir(os.path.dirname(__file__)))
from ml.ldm import LDM

diff_model = torch.load('models/ldm-basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)-0005-100-1400.pkl').to('cuda', dtype=torch.float32)
font_model = torch.load('models/transformer-basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded-14.pkl').to('cuda', dtype=torch.bfloat16)

def sample_glyphs(feature_im : torch.Tensor = None, mask : torch.Tensor = None):
    latent_shape = (1, 26, 2048)
    return diff_model.sample(latent_shape)