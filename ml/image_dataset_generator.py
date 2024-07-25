import pathlib
import csv
from config import operators
from glyph_viz import Visualizer
from tokenizer import Tokenizer
from tablelist_utils import make_non_cumulative, numbers_first
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from PIL import Image


def generate_image_dataset(dataset_name : str, im_pixel_size : tuple, tokenizer : Tokenizer, save_loc : pathlib.Path):
    print("Loading original dataset...")
    with open(f"./fontmakerai/{dataset_name}", 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        dataset_size = len(csv_reader)
        num_glyphs = 70
        assert dataset_size % num_glyphs == 0, f"Dataset must be divisible by number of glyphs ({num_glyphs})"
        
        crop_factor = 1.5
        boundaries = (int((im_pixel_size[0] * (crop_factor - 1)) // 2), int((im_pixel_size[1] * (crop_factor - 1)) // 2))
        ppi = 100
        im_size_inches = ((im_pixel_size[0] * crop_factor) / ppi, (im_pixel_size[1] * crop_factor) / ppi)
        dataset = torch.zeros((dataset_size // num_glyphs, num_glyphs, im_pixel_size[0], im_pixel_size[1]))

        ii = 0
        try:
            for row in csv_reader:
                if '' in row:
                    raise Exception("Cannot have empty cell in dataset")
                else:
                    viz = Visualizer(row)
                    arr = viz.draw(display=False, filename=None, plot_outline=False,
                                plot_control_points=False, return_image=True,
                                bounds=(tokenizer.min_number, tokenizer.max_number),
                                im_size_inches=im_size_inches, center=True)[boundaries[0]:-boundaries[0],boundaries[1]:-boundaries[1],0]
                    dataset[ii // num_glyphs, ii % num_glyphs,:,:] = torch.IntTensor(arr.copy())
                    ii += 1
        except Exception as e:
            print(f"Exception occurred: {e.args[0]}")
            breakpoint()
        
        torch.save(dataset, save_loc)
        print(f"Dataset saved to {save_loc}")


if __name__ == "__main__":
    min_number = -500
    max_number = 500
    pad_token = "<PAD>"
    sos_token = "<SOS>"
    eos_token = "<EOS>"
    tokenizer = Tokenizer(
        min_number=min_number,
        max_number=max_number,
        possible_operators=operators,
        pad_token=pad_token,
        sos_token=sos_token,
        eos_token=eos_token
    )

    im_size_pixels = (128, 128)
    generate_image_dataset(
        dataset_name="47000_fonts_centered_scaled.csv",
        im_pixel_size=im_size_pixels,
        tokenizer=tokenizer,
        save_loc=pathlib.Path(__file__).parent.parent.joinpath(f"47000_images_filtered_{min_number}_{max_number}_{im_size_pixels}.pt")
    )