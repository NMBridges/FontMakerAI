class Tokenizer:
    def __init__(self, min_number : int = -1500, max_number : int = 1500, possible_operators : list[str] = [],
                sos_token : str = '<SOS>', eos_token : str = '<EOS>', pad_token : str = '<PAD>'):

        self.min_number = min_number
        self.max_number = max_number
        self.possible_operators = possible_operators
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.num_tokens = 1 + 1 + 1 + len(possible_operators) + max_number - min_number + 1
        self.map = {
            pad_token: 0,
            sos_token: 1,
            eos_token: 2
        }

        run_val = 3

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

    def reverse_map(self, index : int) -> str:
        if index < 0 or index >= self.num_tokens:
            raise Exception(f"Invalid index. Index must be between {0} and {self.num_tokens-1} (inclusive")
        elif index < 3:
            return [self.pad_token, self.sos_token, self.eos_token][index]
        elif index < 3 + len(self.possible_operators):
            return self.possible_operators[index - 3]
        else:
            return f'{self.min_number + index - (3 + len(self.possible_operators))}'