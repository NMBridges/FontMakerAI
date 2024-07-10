import pathlib
from config import operators
from glyph_viz import Visualizer
from dataset_creator import BucketedDataset
from tokenizer import Tokenizer
from tablelist_utils import make_non_cumulative, numbers_first
from torch.utils.data import DataLoader


def generate_image_dataset(dataset_name : str, tokenizer : Tokenizer, save_loc : pathlib.Path):
    train_tensor_dataset = BucketedDataset(f"./fontmakerai/{dataset_name}", tokenizer, (0,1), cumulative=True)
    train_dataset_size = len(train_tensor_dataset)
    from PIL import Image
    ii = 0
    for seq in train_tensor_dataset:
        padded = [tokenizer.reverse_map(x.item(), use_int=True) for x in seq]
        print(padded)
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
                    plot_control_points=False, return_image=True)[:,:,0]
        im = Image.fromarray(arr)
        im.save(f"./fontmakerai/training_images/Zz{ii}.png")
        ii += 1


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

    generate_image_dataset(
        dataset_name="46918_fonts.csv",
        tokenizer=tokenizer,
        save_loc=pathlib.Path(".")
    )