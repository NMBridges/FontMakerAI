class Tokenizer:
    def __init__(self, min_number : int = -1500, max_number : int = 1500, possible_operators : list[str] = [],
                sos_token : str = '<SOS>', eos_token : str = '<EOS>', pad_token : str = '<PAD>'):

        self.min_number = min_number
        self.max_number = max_number
        self.possible_operators = possible_operators
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.pad2_token = '<PAD2>'

        self.map = {
            pad_token: 0,
            sos_token: 1,
            eos_token: 2,
            self.pad2_token: 3,
        }
        self.special_tokens_len = len(self.map)
        self.num_tokens = self.special_tokens_len + len(possible_operators) + max_number - min_number + 1

        run_val = self.special_tokens_len

        for operator in possible_operators:
            if operator in self.map:
                raise Exception(f"Cannot name operator {operator}; name already used")
            self.map[operator] = run_val
            run_val += 1

        for number in range(min_number, max_number + 1):
            self.map[f'{number}'] = run_val
            run_val += 1

    def __getitem__(self, key : str) -> int:
        return self.map[key]

    def reverse_map(self, index : int, use_int : bool = False) -> str:
        if index < 0 or index >= self.num_tokens:
            raise Exception(f"Invalid index. Index must be between {0} and {self.num_tokens-1} (inclusive")
        elif index < self.special_tokens_len:
            return [self.pad_token, self.sos_token, self.eos_token, self.pad2_token][index]
        elif index < self.special_tokens_len + len(self.possible_operators):
            return self.possible_operators[index - self.special_tokens_len]
        elif use_int:
            return self.min_number + index - (self.special_tokens_len + len(self.possible_operators))
        else:
            return f'{self.min_number + index - (self.special_tokens_len + len(self.possible_operators))}'