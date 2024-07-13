import pathlib
from config import operators
from glyph_viz import Visualizer
from dataset_creator import BucketedDataset
from tokenizer import Tokenizer
from tablelist_utils import make_non_cumulative, numbers_first
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from PIL import Image


def generate_image_dataset(dataset_name : str, im_pixel_size : tuple, tokenizer : Tokenizer, save_loc : pathlib.Path):
    print("Loading original dataset...")
    tensor_dataset = BucketedDataset(f"./fontmakerai/{dataset_name}", tokenizer, (0,-1), cumulative=True)
    dataset_size = len(tensor_dataset)

    crop_factor = 1.5
    boundaries = (int((im_pixel_size[0] * (crop_factor - 1)) // 2), int((im_pixel_size[1] * (crop_factor - 1)) // 2))
    num_channels = 1
    ppi = 100
    im_size_inches = ((im_pixel_size[0] * crop_factor) / ppi, (im_pixel_size[1] * crop_factor) / ppi)
    img_dataset = torch.zeros(
        (
            dataset_size,
            num_channels,
            im_pixel_size[0],
            im_pixel_size[1]
        )
    )

    print("Converting dataset to images...")
    ii = 0
    for seq in tqdm(tensor_dataset):
        try:
            padded = [tokenizer.reverse_map(x.item(), use_int=True) for x in seq]
            if tokenizer.eos_token in padded:
                padded = padded[:padded.index(tokenizer.eos_token)]
            else:
                raise Exception("No EOS token found from dataset")
            seq_toks = numbers_first(
                make_non_cumulative(
                    padded,
                    tokenizer
                ),
                tokenizer,
                return_string=False
            )
            viz = Visualizer(seq_toks)
            arr = viz.draw(display=False, filename=None, plot_outline=False,
                        plot_control_points=False, return_image=True,
                        bounds=(tokenizer.min_number, tokenizer.max_number),
                        im_size_inches=im_size_inches, center=True)[boundaries[0]:-boundaries[0],boundaries[1]:-boundaries[1],0]
            img_dataset[ii,0,:,:] = torch.IntTensor(arr.copy())
            ii += 1
        except Exception as e:
            print(f"Exception occurred: {e.args[0]}")
            breakpoint()

    torch.save(img_dataset, save_loc)

    print(f"Dataset saved to {save_loc}")


if __name__ == "__main__":
    min_number = -1500
    max_number = 1500
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
        dataset_name="46918_fonts.csv",
        im_pixel_size=im_size_pixels,
        tokenizer=tokenizer,
        save_loc=pathlib.Path(__file__).parent.parent.joinpath(f"47000_images_filtered_{min_number}_{max_number}_{im_size_pixels}.pt")
    )