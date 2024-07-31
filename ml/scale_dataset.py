import pathlib
import csv
from config import operators
from glyph_viz import Visualizer
from dataset_creator import BucketedDataset
from tokenizer import Tokenizer
from tablelist_utils import make_non_cumulative, numbers_first, center_and_scale
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from PIL import Image


if __name__ == "__main__":
    dataset_name = "47000_fonts.csv"
    csv_filepath = f"./fontmakerai/{dataset_name}"
    new_csv_filepath = f"./fontmakerai/{dataset_name.split('.')[0]}_centered_scaled.csv"
    
    print("Loading original dataset...")

    min_number = -1500 # doesn't matter
    max_number = 1500 # doesn't matter
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

    with open(new_csv_filepath, 'w', newline='\n', encoding='utf8') as out_csv_file:
        csv_writer = csv.writer(out_csv_file)
        with open(csv_filepath, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for idx, row in enumerate(tqdm(csv_reader)):
                if '' in row:
                    raise Exception("Cannot have empty cell in dataset")
                else:
                    try:
                        trunc_row = center_and_scale(row, tokenizer, return_string=False)
                    except Exception as e:
                        print(idx)
                        breakpoint()
                csv_writer.writerow(trunc_row)