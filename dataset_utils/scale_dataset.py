import pathlib
import csv
from config import operators
from tokenizer import Tokenizer
from tablelist_utils import operator_first, center_and_scale
from tqdm import tqdm


if __name__ == "__main__":
    dataset_name = "basic-35851allchars_filtered.csv"
    csv_filepath = f"./{dataset_name}"
    new_csv_filepath = f"./{dataset_name.split('.')[0]}_centered_scaled.csv"
    
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
                    except Exception as ex:
                        print(idx, ex, row)
                        breakpoint()
                csv_writer.writerow(trunc_row)