import csv
from tokenizer import Tokenizer
import torch


class BucketedDataset(torch.utils.data.Dataset):
    def __init__(self, csv_filepath : str, tokenizer : Tokenizer, bucket_range : tuple[int, int]):
        super(BucketedDataset, self).__init__()
        with open(csv_filepath, 'r', encoding='utf8') as csv_file:
            rows = []
            row_lens = []
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                trunc_row = row[:row.index('')]
                if len(trunc_row) == 0:
                    continue
                
                # Filter out fonts with large offsets to reduce vocab size; remove once model is good enough
                font_good = True
                for x in trunc_row:
                    try:
                        xx = int(x)
                        if xx > tokenizer.max_number or xx < tokenizer.min_number:
                            font_good = False
                            break
                    except:
                        continue
                
                if font_good:
                    rows.append(trunc_row)
                    row_lens.append(len(trunc_row))

            (rows, row_lens) = zip(*sorted(zip(rows, row_lens), key=lambda x : x[1]))
            
            # num_buckets = 10
            break_x = 256 # len(sorted_lens) // num_buckets # The bucket length
            dataset = []
            print(f"Bucket boundaries:\n{row_lens[0]}")
            for i in range(*bucket_range):
                start_idx = i * break_x
                end_idx = (i + 1) * break_x
                bucket = torch.ones((end_idx - start_idx, row_lens[(i + 1) * break_x - 1] + 1)).int() * tokenizer[tokenizer.pad_token]
                print(row_lens[(i + 1) * break_x - 1])
                for idx, row in enumerate(rows[start_idx:end_idx]):
                    for col in range(row_lens[start_idx + idx]):
                        bucket[idx, col] = tokenizer[row[col]]
                    bucket[idx, row_lens[start_idx + idx]] = tokenizer[tokenizer.eos_token]
                dataset.append(bucket)

            self.dataset = dataset

    def __len__(self):
        return sum([dset.shape[0] for dset in self.dataset])

    def __getitem__(self, idx):
        # if idx < 0 or idx >= len(self):
        #     raise Exception(f"idx out of bounds. Must be in [{0},{len(self)})")
        if isinstance(idx, slice):
            new_slice = slice(idx.start % self.dataset[0].shape[0], idx.stop % self.dataset[0].shape[0])
            return self.dataset[idx.start // self.dataset[0].shape[0]][new_slice]
        else:
            return self.dataset[idx // self.dataset[0].shape[0]][idx % self.dataset[0].shape[0]]