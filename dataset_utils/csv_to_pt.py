import csv
import torch
from tokenizer import Tokenizer
from config import operators
from tablelist_utils import use_basic_operators, operator_first
from tqdm import tqdm


def csv_to_pt(csv_filepath : str, pt_filepath : str, tokenizer : Tokenizer, num_glyphs : int, max_length : int = -1):
    with open(csv_filepath, 'r', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        dataset_size = len([0 for _ in reader])
        longest_entry = 2000
        # dataset_size = 0
        # longest_entry = 0
        # flag = True
        # lens = []
        # for idx, row in tqdm(enumerate(reader)):
        #     int_row = [int(el) for el in row if el not in operators]
        #     length_sat = len(row) + 2 <= max_length or max_length == -1
        #     tokens_sat = max(int_row) <= tokenizer.max_number and min(int_row) >= tokenizer.min_number
        #     if length_sat and tokens_sat:
        #         lens.append(len(row) + 2)
        #     else:
        #         flag = False
        #     if (idx + 1) % num_glyphs == 0:
        #         if flag:
        #             dataset_size += num_glyphs
        #             longest_entry = max(longest_entry, max(lens))
        #         flag = True
        #         lens = []
        # print(f"{dataset_size=}")
        # print(f"{longest_entry=}")
    with open(csv_filepath, 'r', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        pt_dataset = torch.zeros((dataset_size, longest_entry), dtype=torch.int16)
        running_idx = 0
        stack = []
        flag = True
        for idx, row in tqdm(enumerate(reader)):
            int_row = [int(el) for el in row if el not in operators]
            length_sat = len(row) + 2 <= max_length or max_length == -1
            tokens_sat = max(int_row) <= tokenizer.max_number and min(int_row) >= tokenizer.min_number
            if length_sat and tokens_sat:
                stack.append([tokenizer.sos_token] + operator_first(row, tokenizer) + [tokenizer.eos_token])
            else:
                flag = False
            if (idx + 1) % num_glyphs == 0:
                if flag:
                    for r in stack:
                        for c, el in enumerate(r):
                            pt_dataset[running_idx, c] = tokenizer[el]
                        running_idx += 1
                flag = True
                stack = []
        torch.save(pt_dataset, pt_filepath)

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

    dataset_name = "basic-30513allchars_filtered_centered_scaled"
    csv_dataset_name = f"{dataset_name}.csv"
    pt_dataset_name = f"{dataset_name}.pt"
    csv_to_pt(f'./{csv_dataset_name}', f'./{pt_dataset_name}', tokenizer, 91, 2000)