import csv
from tokenizer import Tokenizer
from tablelist_utils import operator_first, make_cumulative
import torch
import numpy as np


class BucketedDataset(torch.utils.data.Dataset):
    def __init__(self, csv_filepath : str, tokenizer : Tokenizer, bucket_range : tuple[int, int]):
        super(BucketedDataset, self).__init__()
        with open(csv_filepath, 'r', encoding='utf8') as csv_file:
            rows = []
            row_lens = []
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                if '' in row:
                    raise Exception("Cannot have empty cell in dataset")
                else:
                    trunc_row = make_cumulative(operator_first(row, tokenizer), tokenizer)
                
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

            row_lens_np = np.array(row_lens)
            min_len = 14
            min_idx = np.searchsorted(row_lens_np, min_len, 'left')
            max_len = 1700
            max_idx = np.searchsorted(row_lens_np, max_len, 'right')
            rows = rows[min_idx:max_idx]
            row_lens = row_lens[min_idx:max_idx]
            
            bucket_size = 256
            num_buckets = len(row_lens) // bucket_size
            # bucket_size = len(row_lens) // num_buckets # The bucket length
            
            start_bucket = bucket_range[0]
            if start_bucket < 0:
                start_bucket += num_buckets
            end_bucket = bucket_range[1]
            if end_bucket < 0:
                end_bucket += num_buckets

            print(f"Bucket size: {bucket_size}")
            print(f"Bucket range: ({start_bucket}, {end_bucket})")
            print(f"Bucket boundaries:\n [{row_lens[start_bucket * bucket_size]}", end='')
            dataset = []
            
            for i in range(start_bucket, end_bucket):
                start_idx = i * bucket_size
                end_idx = (i + 1) * bucket_size
                bucket = torch.ones((end_idx - start_idx, row_lens[(i + 1) * bucket_size - 1] + 1)).long() * tokenizer[tokenizer.pad_token]
                print(',', row_lens[(i + 1) * bucket_size - 1], end='')
                for idx, row in enumerate(rows[start_idx:end_idx]):
                    for col in range(row_lens[start_idx + idx]):
                        bucket[idx, col] = tokenizer[row[col]]
                    bucket[idx, row_lens[start_idx + idx]] = tokenizer[tokenizer.eos_token]
                dataset.append(bucket)
            print("]")

            ### For single item overfitting
            # bucket_size = 1
            # for i in range(1):
            #     start_idx = i * bucket_size
            #     end_idx = (i + 1) * bucket_size
            #     bucket = torch.ones((bucket_size, row_lens[(i + 1) * bucket_size - 1] + 1)).long() * tokenizer[tokenizer.pad_token]
            #     print(',', row_lens[(i + 1) * bucket_size - 1], end='')
            #     for idx, row in enumerate(rows[start_idx:end_idx]):
            #         for col in range(row_lens[start_idx + idx]):
            #             bucket[idx, col] = tokenizer[row[col]]
            #             print(row[col])
            #         bucket[idx, row_lens[start_idx + idx]] = tokenizer[tokenizer.eos_token]
            #     dataset.append(bucket)
            # print("]")

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