import csv
from config import operators
from tokenizer import Tokenizer
from tablelist_utils import make_cumulative, operator_first
from tqdm import tqdm
from torch.utils.data import DataLoader


if __name__ == "__main__":
    dataset_name = "basic-35851allchars.csv"
    csv_filepath = f"./{dataset_name}"
    new_csv_filepath = f"./{dataset_name.split('.')[0]}_2.csv"

    
    print("Loading original dataset...")

    bad_operators = [
        "hstem",
        "vstem",
        "hstemhm",
        "vstemhm",
        "hintmask",
        "cntrmask",
        "callsubr",
        "callgsubr",
        "vsindex",
        "blend"
    ]

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

    max_length = 2000
    min_length = 8
    num_glyphs = 91

    def queue_good(queue):
        for r in queue:
            if len(r) < min_length or len(r) + 2 > max_length:
                return False
            for bop in bad_operators:
                if bop in r:
                    return False
            try:
                make_cumulative(operator_first(r, tokenizer), tokenizer)
            except Exception as e:
                print(e.args[0])
                return False
        return True
        

    queue = None

    with open(new_csv_filepath, 'w', newline='\n', encoding='utf8') as out_csv_file:
        csv_writer = csv.writer(out_csv_file)
        with open(csv_filepath, 'r', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for idx, row in enumerate(tqdm(csv_reader)):
                if idx % num_glyphs == 0:
                    if queue is not None:
                        if queue_good(queue):
                            for r in queue:
                                csv_writer.writerow(r)
                    queue = [row]
                else:
                    queue.append(row)
            if queue is not None and queue_good(queue):
                for r in queue:
                    csv_writer.writerow(r)