import csv
from tokenizer import Tokenizer
import torch


def gen_dataset(filepath : str, max_offset_magnitude : int = 1500) -> torch.Tensor:
    with open(filepath, 'r', encoding='utf8') as csv_file:
        rows = []
        row_lens = []
        csv_reader = csv.reader(csv_file)

        min_number = -max_offset_magnitude
        max_number = max_offset_magnitude

        for row in csv_reader:
            trunc_row = row[:row.index('')]
            if len(trunc_row) == 0:
                continue
            
            # Filter out fonts with large offsets to reduce vocab size; remove once model is good enough
            font_good = True
            for x in trunc_row:
                try:
                    xx = int(x)
                    if xx > max_number or xx < min_number:
                        font_good = False
                        break
                except:
                    continue
            
            if font_good:
                rows.append(trunc_row)
                row_lens.append(len(trunc_row))

        tokenizer = Tokenizer(
            min_number=min_number,
            max_number=max_number,
            possible_operators=[
                # Should be 30
                "rmoveto",
                "hmoveto",
                "vmoveto",
                "rlineto",
                "hlineto",
                "vlineto",
                "rrcurveto",
                "hhcurveto",
                "vvcurveto",
                "hvcurveto",
                "vhcurveto",
                "rcurveline",
                "rlinecurve",
                "flex",
                "hflex",
                "hflex1",
                "flex1",
                "hstem",
                "vstem",
                "hstemhm",
                "vstemhm",
                "hintmask",
                "cntrmask",
                "callsubr",
                "callgsubr",
                "vsindex",
                "blend",
                "endchar"
            ]
        )
        (sorted_rows, sorted_lens) = zip(*sorted(zip(rows, row_lens), key=lambda x : x[1]))
        
        num_buckets = 10
        break_x = 256 # len(sorted_lens) // num_buckets
        dataset = []
        print(f"Bucket boundaries:\n{sorted_lens[0]}")
        for i in range(num_buckets):
            start_idx = i * break_x
            end_idx = (i + 1) * break_x
            bucket = torch.ones((end_idx - start_idx, sorted_lens[(i + 1) * break_x - 1])).int() * tokenizer[tokenizer.pad_token]
            print(sorted_lens[(i + 1) * break_x - 1])
            for idx, row in enumerate(sorted_rows[start_idx:end_idx]):
                for col in range(sorted_lens[start_idx + idx]):
                    bucket[idx, col] = tokenizer[row[col]]
            dataset.append(bucket)

        return dataset

if __name__ == "__main__":
    dataset = (gen_dataset('./ml/data_no_subr.csv'))
    
    breakpoint()