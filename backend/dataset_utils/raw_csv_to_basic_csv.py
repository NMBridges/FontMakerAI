import csv
from tokenizer import Tokenizer
from config import operators
from tablelist_utils import use_basic_operators
from tqdm import tqdm


def raw_csv_to_basic_csv(in_filepath : str, out_filepath : str, tokenizer : Tokenizer):
    with open(in_filepath, 'r', encoding='utf8') as in_file:
        reader = csv.reader(in_file)
        with open(out_filepath, 'w', encoding='utf8') as out_file:
            writer = csv.writer(out_file, delimiter=',')
            stack = []
            flag = True
            for idx, row in tqdm(enumerate(reader)):
                if '' in row:
                    raise Exception("Cannot have empty cell in dataset")
                else:
                    try:
                        trunc_row = use_basic_operators(row, tokenizer)
                        if len(trunc_row) < 12:
                            flag = False
                        else:
                            stack.append(trunc_row)
                    except Exception as e:
                        flag = False
                    if (idx+1) % 91 == 0:
                        if flag:
                            for r in stack:
                                writer.writerow(r)
                        else:
                            flag = True
                        stack = []

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

    
    dataset_name = "35851allchars.csv"
    raw_csv_to_basic_csv(f'./{dataset_name}', f'./basic-{dataset_name}', tokenizer)