import pathlib
import csv
from config import operators, bad_operators
from glyph_viz import Visualizer
from tokenizer import Tokenizer
from tablelist_utils import make_non_cumulative, numbers_first, operator_first, center_and_scale
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from PIL import Image


def tablelist_to_image(tablelist : list, im_size_inches : tuple, boundaries : tuple):
    '''
    Takes a raw tablelist (numbers first, non-cumulative) and image size specifications,
    and returns the numpy array for the image.
    
    Parameters:
    -----------
    tablelist (list[str]): the raw tablelist
    im_size_inches (tuple[float, float]): the desired image dimensions in inches
    boundaries (tuple[int, int]): the boundaries to crop to

    Returns:
    --------
    torch.Tensor: the image corresponding to the tablelist
    '''
    num_row = center_and_scale(tablelist, tokenizer, return_string=False)
    viz = Visualizer(num_row)
    arr = viz.draw(display=False, filename=None, plot_outline=False,
                plot_control_points=False, return_image=True,
                bounds=(tokenizer.min_number, tokenizer.max_number),
                im_size_inches=im_size_inches, center=True)[boundaries[0]:-boundaries[0],boundaries[1]:-boundaries[1],0]
    return torch.IntTensor(arr.copy())


def queue_good(queue : list, im_size_pixels : tuple, im_size_inches : tuple, boundaries : tuple):
    '''
    Takes in a list of tablelists and returns their respective images if they are valid, and returns None if not.r

    Parameters:
    -----------
    queue (list[list[str]]): the queue of tablelists
    im_size_pixels (tuple[int, int]): the image size in pixels
    im_size_inches (tuple[float, float]): the desired image dimensions in inches
    boundaries (tuple[int, int]): the boundaries to crop to

    Returns:
    --------
    torch.Tensor: the images corresponding to the queue of tablelists
    '''
    images = torch.zeros((len(queue), im_size_pixels[0], im_size_pixels[1]), dtype=torch.int8)
    for idx, r in enumerate(queue):
        if len(r) < 8:
            return None
        for bop in bad_operators:
            if bop in r:
                return None
        try:
            images[idx] = tablelist_to_image(r, im_size_inches, boundaries)
        except Exception as e:
            print(e.args[0])
            return None
    return images


def generate_image_dataset(dataset_name : str, im_pixel_size : tuple, tokenizer : Tokenizer, save_loc : pathlib.Path):
    '''
    Takes in a dataset of num_glyphs * num_fonts tablelists, where each index (mod num_glyphs) corresponds to the
    same glyph, and all indices (// num_glyphs) that are the same are members of the same font.
        e.g. num_glyphs = 3
            TL_1
            TL_2
            TL_3
            TL_4
            TL_5
            TL_6
        here, TL_1 and TL_4 correspond to the same glyph, and likewise for TL_2/TL_5, as well as TL_3/TL_6
        TL_1, TL_2, and TL_3 correspond to font 1, while TL_4, TL_5, and TL_6 correspond to font 2

    Parameters:
    -----------
    dataset_name (str): the name of the dataset (.csv included) within the fontmakerai folder
    im_pixel_size (tuple[int, int]): the desired resolution of the output image
    tokenizer (Tokenizer): the tokenizer to use
    save_loc (pathlib.Path): the path to the location to store the output dataset
    '''
    ### Grabs the dataset length
    print("Loading original dataset...")
    with open(f"./fontmakerai/{dataset_name}", 'r', encoding='utf8') as csv_file:
        dataset_size = len(csv_file.readlines())
        print(f"{dataset_size=}")

    ### Builds the image dataset
    with open(f"./fontmakerai/{dataset_name}", 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        num_glyphs = 91
        assert dataset_size % num_glyphs == 0, f"Dataset must be divisible by number of glyphs ({num_glyphs})"
        
        crop_factor = 1.5
        boundaries = (int((im_pixel_size[0] * (crop_factor - 1)) // 2), int((im_pixel_size[1] * (crop_factor - 1)) // 2))
        ppi = 100
        im_size_inches = ((im_pixel_size[0] * crop_factor) / ppi, (im_pixel_size[1] * crop_factor) / ppi)
        dataset = torch.zeros((dataset_size // num_glyphs, num_glyphs, im_pixel_size[0], im_pixel_size[1]), dtype=torch.int8)

        ### Has queue to ensure every glyph for a font is valid before adding it to the dataset
        row_queue = None
        font_count = 0
        for idx, row in enumerate(tqdm(csv_reader)):
            if idx % num_glyphs == 0:
                if row_queue is not None:
                    # Deal with old queue
                    queue_check_output = queue_good(row_queue, im_size_pixels, im_size_inches, boundaries)
                    if queue_check_output is not None:
                        dataset[font_count,:,:,:] = queue_check_output
                        font_count += 1
                row_queue = [row]
            else:
                row_queue += [row]
        
        ### Save the dataset to file
        torch.save(dataset[:font_count], save_loc)
        print(f"Dataset of length {font_count} saved to {save_loc}")


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

    im_size_pixels = (64, 64)
    generate_image_dataset(
        dataset_name="35851allchars.csv",
        im_pixel_size=im_size_pixels,
        tokenizer=tokenizer,
        save_loc=pathlib.Path(__file__).parent.parent.joinpath(f"35851allchars_centered_scaled_{min_number}_{max_number}_{im_size_pixels}.pt")
    )