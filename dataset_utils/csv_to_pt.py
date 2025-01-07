import csv
import torch
from tokenizer import Tokenizer
from config import operators
from tablelist_utils import use_basic_operators, operator_first, make_cumulative, pad_tablelist
from tqdm import tqdm


def csv_to_pt(csv_filepath : str, pt_filepath : str, tokenizer : Tokenizer, num_glyphs : int, max_length : int = -1):
    with open(csv_filepath, 'r', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        dataset_size = len([0 for _ in reader])
        print(f"{dataset_size=}")
    with open(csv_filepath, 'r', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        pt_dataset = torch.zeros((dataset_size, max_length), dtype=torch.int16)
        for idx, row in tqdm(enumerate(reader)):
            rr = pad_tablelist(make_cumulative(operator_first(row, tokenizer), tokenizer), tokenizer, return_string=True)
            # rr = make_cumulative(operator_first(row, tokenizer), tokenizer)
            int_row = [int(el) for el in rr if (el not in operators) and (el != tokenizer.pad2_token)]
            length_sat = len(rr) + 2 <= max_length or max_length == -1
            tokens_sat = max(int_row) <= tokenizer.max_number and min(int_row) >= tokenizer.min_number
            if not length_sat or not tokens_sat:
                print("fuckup!", row, rr)
            for c, el in enumerate(rr + [tokenizer.eos_token]):
                pt_dataset[idx, c] = tokenizer[el]
        torch.save(pt_dataset, pt_filepath)

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

    dataset_name = "basic-33698allchars_centered_scaled_sorted_filtered"
    csv_dataset_name = f"{dataset_name}.csv"
    pt_dataset_name = f"{dataset_name}_cumulative_padded.pt"
    csv_to_pt(f'./{csv_dataset_name}', f'./{pt_dataset_name}', tokenizer, 26, 5880)
    # csv_to_pt(f'./{csv_dataset_name}', f'./{pt_dataset_name}', tokenizer, 26, 2000)