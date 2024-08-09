from tokenizer import Tokenizer
from config import operators


def operator_first(tablelist : list, tokenizer : Tokenizer) -> list:
    '''
    Reorders a tablelist such that the operators are before their numeric arguments.

    Parameters:
    -----------
    tablelist (list[str]): the tablelist to reorder

    Returns:
    --------
    list: the reordered tablelist
    '''
    out_list = [0] * len(tablelist)
    op_idx = 0
    for col in range(len(tablelist)):
        if tablelist[col] not in tokenizer.possible_operators:
            out_list[col + 1] = tablelist[col]
        else:
            out_list[op_idx] = tablelist[col]
            op_idx = col + 1
    return out_list


def numbers_first(tablelist : list, tokenizer : Tokenizer, return_string : bool = True) -> list:
    '''
    Reorders a tablelist such that the operators are after their numeric arguments.

    Parameters:
    -----------
    tablelist (list[str]): the tablelist to reorder
    return_string (bool): whether or not to return the numbers in string form

    Returns:
    --------
    list: the reordered tablelist
    '''
    ops = []
    nums = []
    for col in range(len(tablelist)):
        if tablelist[col] in tokenizer.possible_operators:
            ops.append(tablelist[col])
            nums.append([])
        elif tablelist[col] != tokenizer.eos_token and tablelist[col] != tokenizer.sos_token and tablelist[col] != tokenizer.pad_token:
            if len(nums) == 0:
                raise Exception("Generated 'table list' cannot start with a non-operator")
            if return_string:
                nums[-1].append(str(tablelist[col]))
            else:
                nums[-1].append(int(tablelist[col]))
    out_list = []
    i = 0
    j = 0
    while i < len(ops) or j < len(nums):
        if j < len(nums):
            out_list += nums[j]
            j += 1
        if i < len(ops):
            out_list.append(ops[i])
            i += 1
    return out_list


def make_cumulative(tablelist : list, tokenizer : Tokenizer, return_string : bool = True, return_stats : bool = False) -> list:
    '''
    Alters an inverted tablelist (operator first) such that numeric arguments are cumulative

    Parameters:
    -----------
    tablelist (list[str]): the tablelist to alter (operators are first); must start with operator
    tokenizer (Tokenizer): the tokenizer
    return_string (bool): whether or not to return the numbers in string form
    return_stats (bool): whether or not to return stats on the mean, min, and max for each axis

    Returns:
    --------
    list: the altered tablelist
    (optional) tuple[list]: the list of cX and cY
    '''
    out_list = []

    if len(tablelist) == 0 or tablelist[0] not in tokenizer.possible_operators:
        raise Exception("Tablelist must start with operator")

    cX_list = []
    cY_list = []
    def stats_X(new_cX):
        cX_list.append(new_cX)
    def stats_Y(new_cY):
        cY_list.append(new_cY)
        
    cX = 0
    cY = 0
    running_idx = 0
    while running_idx < len(tablelist):
        operator = tablelist[running_idx]
        out_list.append(operator)
        op_idx = running_idx
        running_idx += 1
        while running_idx < len(tablelist) and tablelist[running_idx] not in tokenizer.possible_operators \
            and tablelist[running_idx] != tokenizer.eos_token and tablelist[running_idx] != tokenizer.sos_token \
                and tablelist[running_idx] != tokenizer.pad_token:
            running_idx += 1

        numbers = [int(num) for num in tablelist[op_idx+1:running_idx]]

        if len(numbers) == 0 and operator != "endchar":
            continue

        if operator == "rmoveto":
            # First two numbers are coordinates; (optional) third is width
            if len(numbers) == 2:
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[1]
                out_list.append(cY)
                stats_Y(cY)
            elif len(numbers) == 3 and running_idx == 4: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                cX += numbers[1]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[2]
                out_list.append(cY)
                stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "hmoveto":
            # First number is a coordinate; (optional) second is width
            if len(numbers) == 1:
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                stats_Y(cY)
            elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                cX += numbers[1]
                out_list.append(cX)
                stats_X(cX)
                stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vmoveto":
            # First number is a coordinate; (optional) second is width
            if len(numbers) == 1:
                cY += numbers[0]
                out_list.append(cY)
                stats_Y(cY)
                stats_X(cX)
            elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                cY += numbers[1]
                out_list.append(cY)
                stats_Y(cY)
                stats_X(cX)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rlineto":
            rep_size = 2 # Repeat size
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by two-dimension offset
                    cX += numbers[rep_size * num_dx]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 1]
                    out_list.append(cY)
                    stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "hlineto":
            rep_size = 2
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by x-dimension offset
                    cX += numbers[rep_size * num_dx]
                    out_list.append(cX)
                    stats_X(cX)
                    # Extend path by y-dimension offset
                    cY += numbers[rep_size * num_dx + 1]
                    out_list.append(cY)
                    stats_Y(cY)
            elif len(numbers) % rep_size == 1 and len(numbers) > 0:
                # Extend path by x-dimension offset
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by y-dimension offset
                    cY += numbers[rep_size * num_dx + 1]
                    out_list.append(cY)
                    stats_Y(cY)
                    # Extend path by x-dimension offset
                    cX += numbers[rep_size * num_dx + 2]
                    out_list.append(cX)
                    stats_X(cX)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "vlineto":
            rep_size = 2
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by y-dimension offset
                    cY += numbers[rep_size * num_dx]
                    out_list.append(cY)
                    stats_Y(cY)
                    # Extend path by x-dimension offset
                    cX += numbers[rep_size * num_dx + 1]
                    out_list.append(cX)
                    stats_X(cX)
            elif len(numbers) % rep_size == 1 and len(numbers) > 0:
                # Extend path by y-dimension offset
                cY += numbers[0]
                out_list.append(cY)
                stats_Y(cY)
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by x-dimension offset
                    cX += numbers[rep_size * num_dx + 1]
                    out_list.append(cX)
                    stats_X(cX)
                    # Extend path by y-dimension offset
                    cY += numbers[rep_size * num_dx + 2]
                    out_list.append(cY)
                    stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rrcurveto":
            rep_size = 6
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    cX += numbers[rep_size * num_dx]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 1]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 2]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 3]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 4]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 5]
                    out_list.append(cY)
                    stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hhcurveto":
            rep_size = 4
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    cX += numbers[rep_size * num_dx]
                    out_list.append(cX)
                    stats_X(cX)
                    cX += numbers[rep_size * num_dx + 1]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 2]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 3]
                    out_list.append(cX)
                    stats_X(cX)
            elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                cY += numbers[0]
                out_list.append(cY)
                stats_Y(cY)
                for num_dx in range(len(numbers) // rep_size):
                    cX += numbers[rep_size * num_dx + 1]
                    out_list.append(cX)
                    stats_X(cX)
                    cX += numbers[rep_size * num_dx + 2]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 3]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 4]
                    out_list.append(cX)
                    stats_X(cX)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vvcurveto":
            rep_size = 4
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    cY += numbers[rep_size * num_dx]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 1]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 2]
                    out_list.append(cY)
                    stats_Y(cY)
                    cY += numbers[rep_size * num_dx + 3]
                    out_list.append(cY)
                    stats_Y(cY)
            elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                for num_dx in range(len(numbers) // rep_size):
                    cY += numbers[rep_size * num_dx + 1]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 2]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 3]
                    out_list.append(cY)
                    stats_Y(cY)
                    cY += numbers[rep_size * num_dx + 4]
                    out_list.append(cY)
                    stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hvcurveto":
            rep_size = 8
            if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                for num_dx in range(len(numbers) // rep_size):
                    cX += numbers[rep_size * num_dx]
                    out_list.append(cX)
                    stats_X(cX)
                    cX += numbers[rep_size * num_dx + 1]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 2]
                    out_list.append(cY)
                    stats_Y(cY)
                    cY += numbers[rep_size * num_dx + 3]
                    out_list.append(cY)
                    stats_Y(cY)
                    cY += numbers[rep_size * num_dx + 4]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 5]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 6]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 7]
                    out_list.append(cX)
                    stats_X(cX)
                if len(numbers) % rep_size == 1:
                    cY += numbers[-1]
                    out_list.append(cY)
                    stats_Y(cY)
            elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                cX += numbers[1]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[2]
                out_list.append(cY)
                stats_Y(cY)
                cY += numbers[3]
                out_list.append(cY)
                stats_Y(cY)
                for num_dx in range(len(numbers) // rep_size):
                    cY += numbers[rep_size * num_dx + 4]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 5]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 6]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 7]
                    out_list.append(cX)
                    stats_X(cX)
                    cX += numbers[rep_size * num_dx + 8]
                    out_list.append(cX)
                    stats_X(cX)
                    cX += numbers[rep_size * num_dx + 9]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 10]
                    out_list.append(cY)
                    stats_Y(cY)
                    cY += numbers[rep_size * num_dx + 11]
                    out_list.append(cY)
                    stats_Y(cY)
                if len(numbers) % rep_size == 5:
                    cX += numbers[-1]
                    out_list.append(cX)
                    stats_X(cX)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vhcurveto":
            rep_size = 8
            if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                for num_dx in range(len(numbers) // rep_size):
                    cY += numbers[rep_size * num_dx]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 1]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 2]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 3]
                    out_list.append(cX)
                    stats_X(cX)
                    cX += numbers[rep_size * num_dx + 4]
                    out_list.append(cX)
                    stats_X(cX)
                    cX += numbers[rep_size * num_dx + 5]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 6]
                    out_list.append(cY)
                    stats_Y(cY)
                    cY += numbers[rep_size * num_dx + 7]
                    out_list.append(cY)
                    stats_Y(cY)
                if len(numbers) % rep_size == 1:
                    cX += numbers[-1]
                    out_list.append(cX)
                    stats_X(cX)
            elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                cY += numbers[0]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[1]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[2]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[3]
                out_list.append(cX)
                stats_X(cX)
                for num_dx in range(len(numbers) // rep_size):
                    cX += numbers[rep_size * num_dx + 4]
                    out_list.append(cX)
                    stats_X(cX)
                    cX += numbers[rep_size * num_dx + 5]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 6]
                    out_list.append(cY)
                    stats_Y(cY)
                    cY += numbers[rep_size * num_dx + 7]
                    out_list.append(cY)
                    stats_Y(cY)
                    cY += numbers[rep_size * num_dx + 8]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 9]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 10]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 11]
                    out_list.append(cX)
                    stats_X(cX)
                if len(numbers) % rep_size == 5:
                    cY += numbers[-1]
                    out_list.append(cY)
                    stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                
        elif operator == "rcurveline":
            rep_size = 6
            if len(numbers) % rep_size == 2 and len(numbers) > 2:
                for num_dx in range(len(numbers) // rep_size):
                    cX += numbers[rep_size * num_dx]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 1]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 2]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 3]
                    out_list.append(cY)
                    stats_Y(cY)
                    cX += numbers[rep_size * num_dx + 4]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 5]
                    out_list.append(cY)
                    stats_Y(cY)
                cX += numbers[-2]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[-1]
                out_list.append(cY)
                stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rlinecurve":
            rep_size = 2 # Repeat size
            if len(numbers) % rep_size == 0 and len(numbers) > 6:
                for num_dx in range((len(numbers) - 6) // rep_size):
                    # Extend path by two-dimension offset
                    cX += numbers[rep_size * num_dx]
                    out_list.append(cX)
                    stats_X(cX)
                    cY += numbers[rep_size * num_dx + 1]
                    out_list.append(cY)
                    stats_Y(cY)
                cX += numbers[-6]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[-5]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[-4]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[-3]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[-2]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[-1]
                out_list.append(cY)
                stats_Y(cY)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "flex":
            if len(numbers) == 13:
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[1]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[2]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[3]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[4]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[5]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[6]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[7]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[8]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[9]
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[10]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[11]
                out_list.append(cY)
                stats_Y(cY)
                fd = numbers[12]
                out_list.append(fd)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hflex":
            if len(numbers) == 7:
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                # cY += 0        # d1
                cX += numbers[1]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[2] # d2
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[3]
                out_list.append(cX)
                stats_X(cX)
                # cY += 0        # d3
                cX += numbers[4]
                out_list.append(cX)
                stats_X(cX)
                # cY += 0        # d4
                cX += numbers[5]
                out_list.append(cX)
                stats_X(cX)
                cY -= numbers[2] # d5 -- note this is reusing numbers[2] to return to same Y value
                # DO NOT ADD TO LIST ^^^
                cX += numbers[6]
                out_list.append(cX)
                stats_X(cX)
                # cY += 0        # d6
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                
        elif operator == "hflex1":
            if len(numbers) == 9:
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[1] # d1
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[2]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[3] # d2
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[4]
                out_list.append(cX)
                stats_X(cX)
                # cY += 0        # d3
                cX += numbers[5]
                out_list.append(cX)
                stats_X(cX)
                # cY += 0        # d4
                cX += numbers[6]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[7] # d5
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[8]
                out_list.append(cX)
                stats_X(cX)
                cY -= numbers[1] + numbers[3] + numbers[7] # d6 -- note this reuses numbers to return to same Y value
                # DO NOT ADD TO LIST ^^^
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "flex1":
            if len(numbers) == 11:
                # TODO: Support Bezier curves
                t_X = cX # temp X
                t_Y = cY # temp Y
                cX += numbers[0]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[1] # d1
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[2]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[3] # d2
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[4]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[5]  # d3
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[6]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[7] # d4
                out_list.append(cY)
                stats_Y(cY)
                cX += numbers[8]
                out_list.append(cX)
                stats_X(cX)
                cY += numbers[9] # d5
                out_list.append(cY)
                stats_Y(cY)
                if abs(cX - t_X) > abs(cY - t_Y):
                    cX = numbers[10]
                    out_list.append(cX)
                    stats_X(cX)
                    cY = t_Y
                else:
                    cX = t_X
                    cY = numbers[10]
                    out_list.append(cY)
                    stats_Y(cY)
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hstem":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "vstem":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "hstemhm":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "vstemhm":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "hintmask":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "cntrmask":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "callsubr":
            raise Exception("Operator not implemented")
        elif operator == "callgsubr":
            raise Exception("Operator not implemented")
        elif operator == "vsindex":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "blend":
            # Not in any of the fonts
            raise Exception("Operator not implemented")

        elif operator == "endchar":
            break

        else:
            raise Exception(f"Operator '{operator}' not found")

    if return_string:
        if return_stats:
            return [str(item) for item in out_list], (cX_list, cY_list)
        else:
            return [str(item) for item in out_list]
    else:
        if return_stats:
            return out_list, (cX_list, cY_list)
        else:
            return out_list


def make_non_cumulative(tablelist : list, tokenizer : Tokenizer, return_string : bool = True) -> list:
    '''
    Alters an inverted tablelist (operator first) such that numeric arguments are non-cumulative

    Parameters:
    -----------
    tablelist (list[str]): the tablelist to alter (operators are first); must start with operator
    tokenizer (Tokenizer): the tokenizer
    return_string (bool): whether or not to return the numbers in string form

    Returns:
    --------
    list: the altered tablelist
    '''
    out_list = []

    if len(tablelist) == 0 or tablelist[0] not in tokenizer.possible_operators:
        raise Exception("Tablelist must start with operator")

    cX = 0
    cY = 0
    running_idx = 0
    while running_idx < len(tablelist):
        operator = tablelist[running_idx]
        out_list.append(operator)
        op_idx = running_idx
        running_idx += 1
        while running_idx < len(tablelist) and tablelist[running_idx] not in tokenizer.possible_operators \
            and tablelist[running_idx] != tokenizer.eos_token and tablelist[running_idx] != tokenizer.sos_token \
                and tablelist[running_idx] != tokenizer.pad_token:
            running_idx += 1

        numbers = [int(num) for num in tablelist[op_idx+1:running_idx]]

        if len(numbers) == 0 and operator != "endchar":
            continue

        if operator == "rmoveto":
            # First two numbers are coordinates; (optional) third is width
            if len(numbers) == 2:
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
                out_list.append(numbers[1] - cY)
                cY = numbers[1]
            elif len(numbers) == 3 and running_idx == 4: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                out_list.append(numbers[1] - cX)
                cX = numbers[1]
                out_list.append(numbers[2] - cY)
                cY = numbers[2]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "hmoveto":
            # First number is a coordinate; (optional) second is width
            if len(numbers) == 1:
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
            elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                out_list.append(numbers[1] - cX)
                cX = numbers[1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vmoveto":
            # First number is a coordinate; (optional) second is width
            if len(numbers) == 1:
                out_list.append(numbers[0] - cY)
                cY = numbers[0]
            elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                out_list.append(numbers[1] - cY)
                cY = numbers[1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rlineto":
            rep_size = 2 # Repeat size
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by two-dimension offset
                    out_list.append(numbers[rep_size * num_dx] - cX)
                    cX = numbers[rep_size * num_dx]
                    out_list.append(numbers[rep_size * num_dx + 1] - cY)
                    cY = numbers[rep_size * num_dx + 1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "hlineto":
            rep_size = 2
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by x-dimension offset
                    out_list.append(numbers[rep_size * num_dx] - cX)
                    cX = numbers[rep_size * num_dx]
                    # Extend path by y-dimension offset
                    out_list.append(numbers[rep_size * num_dx + 1] - cY)
                    cY = numbers[rep_size * num_dx + 1]
            elif len(numbers) % rep_size == 1 and len(numbers) > 0:
                # Extend path by x-dimension offset
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by y-dimension offset
                    out_list.append(numbers[rep_size * num_dx + 1] - cY)
                    cY = numbers[rep_size * num_dx + 1]
                    # Extend path by x-dimension offset
                    out_list.append(numbers[rep_size * num_dx + 2] - cX)
                    cX = numbers[rep_size * num_dx + 2]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "vlineto":
            rep_size = 2
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by y-dimension offset
                    out_list.append(numbers[rep_size * num_dx] - cY)
                    cY = numbers[rep_size * num_dx]
                    # Extend path by x-dimension offset
                    out_list.append(numbers[rep_size * num_dx + 1] - cX)
                    cX = numbers[rep_size * num_dx + 1]
            elif len(numbers) % rep_size == 1 and len(numbers) > 0:
                # Extend path by y-dimension offset
                out_list.append(numbers[0] - cY)
                cY = numbers[0]
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by x-dimension offset
                    out_list.append(numbers[rep_size * num_dx + 1] - cX)
                    cX = numbers[rep_size * num_dx + 1]
                    # Extend path by y-dimension offset
                    out_list.append(numbers[rep_size * num_dx + 2] - cY)
                    cY = numbers[rep_size * num_dx + 2]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rrcurveto":
            rep_size = 6
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx] - cX)
                    cX = numbers[rep_size * num_dx]
                    out_list.append(numbers[rep_size * num_dx + 1] - cY)
                    cY = numbers[rep_size * num_dx + 1]
                    out_list.append(numbers[rep_size * num_dx + 2] - cX)
                    cX = numbers[rep_size * num_dx + 2]
                    out_list.append(numbers[rep_size * num_dx + 3] - cY)
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(numbers[rep_size * num_dx + 4] - cX)
                    cX = numbers[rep_size * num_dx + 4]
                    out_list.append(numbers[rep_size * num_dx + 5] - cY)
                    cY = numbers[rep_size * num_dx + 5]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hhcurveto":
            rep_size = 4
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx] - cX)
                    cX = numbers[rep_size * num_dx]
                    out_list.append(numbers[rep_size * num_dx + 1] - cX)
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(numbers[rep_size * num_dx + 2] - cY)
                    cY = numbers[rep_size * num_dx + 2]
                    out_list.append(numbers[rep_size * num_dx + 3] - cX)
                    cX = numbers[rep_size * num_dx + 3]
            elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                out_list.append(numbers[0] - cY)
                cY = numbers[0]
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx + 1] - cX)
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(numbers[rep_size * num_dx + 2] - cX)
                    cX = numbers[rep_size * num_dx + 2]
                    out_list.append(numbers[rep_size * num_dx + 3] - cY)
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(numbers[rep_size * num_dx + 4] - cX)
                    cX = numbers[rep_size * num_dx + 4]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vvcurveto":
            rep_size = 4
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx] - cY)
                    cY = numbers[rep_size * num_dx]
                    out_list.append(numbers[rep_size * num_dx + 1] - cX)
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(numbers[rep_size * num_dx + 2] - cY)
                    cY = numbers[rep_size * num_dx + 2]
                    out_list.append(numbers[rep_size * num_dx + 3] - cY)
                    cY = numbers[rep_size * num_dx + 3]
            elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx + 1] - cY)
                    cY = numbers[rep_size * num_dx + 1]
                    out_list.append(numbers[rep_size * num_dx + 2] - cX)
                    cX = numbers[rep_size * num_dx + 2]
                    out_list.append(numbers[rep_size * num_dx + 3] - cY)
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(numbers[rep_size * num_dx + 4] - cY)
                    cY = numbers[rep_size * num_dx + 4]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hvcurveto":
            rep_size = 8
            if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx] - cX)
                    cX = numbers[rep_size * num_dx]
                    out_list.append(numbers[rep_size * num_dx + 1] - cX)
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(numbers[rep_size * num_dx + 2] - cY)
                    cY = numbers[rep_size * num_dx + 2]
                    out_list.append(numbers[rep_size * num_dx + 3] - cY)
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(numbers[rep_size * num_dx + 4] - cY)
                    cY = numbers[rep_size * num_dx + 4]
                    out_list.append(numbers[rep_size * num_dx + 5] - cX)
                    cX = numbers[rep_size * num_dx + 5]
                    out_list.append(numbers[rep_size * num_dx + 6] - cY)
                    cY = numbers[rep_size * num_dx + 6]
                    out_list.append(numbers[rep_size * num_dx + 7] - cX)
                    cX = numbers[rep_size * num_dx + 7]
                if len(numbers) % rep_size == 1:
                    out_list.append(numbers[-1] - cY)
                    cY = numbers[-1]
            elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
                out_list.append(numbers[1] - cX)
                cX = numbers[1]
                out_list.append(numbers[2] - cY)
                cY = numbers[2]
                out_list.append(numbers[3] - cY)
                cY = numbers[3]
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx + 4] - cY)
                    cY = numbers[rep_size * num_dx + 4]
                    out_list.append(numbers[rep_size * num_dx + 5] - cX)
                    cX = numbers[rep_size * num_dx + 5]
                    out_list.append(numbers[rep_size * num_dx + 6] - cY)
                    cY = numbers[rep_size * num_dx + 6]
                    out_list.append(numbers[rep_size * num_dx + 7] - cX)
                    cX = numbers[rep_size * num_dx + 7]
                    out_list.append(numbers[rep_size * num_dx + 8] - cX)
                    cX = numbers[rep_size * num_dx + 8]
                    out_list.append(numbers[rep_size * num_dx + 9] - cX)
                    cX = numbers[rep_size * num_dx + 9]
                    out_list.append(numbers[rep_size * num_dx + 10] - cY)
                    cY = numbers[rep_size * num_dx + 10]
                    out_list.append(numbers[rep_size * num_dx + 11] - cY)
                    cY = numbers[rep_size * num_dx + 11]
                if len(numbers) % rep_size == 5:
                    out_list.append(numbers[-1] - cX)
                    cX = numbers[-1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vhcurveto":
            rep_size = 8
            if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx] - cY)
                    cY = numbers[rep_size * num_dx]
                    out_list.append(numbers[rep_size * num_dx + 1] - cX)
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(numbers[rep_size * num_dx + 2] - cY)
                    cY = numbers[rep_size * num_dx + 2]
                    out_list.append(numbers[rep_size * num_dx + 3] - cX)
                    cX = numbers[rep_size * num_dx + 3]
                    out_list.append(numbers[rep_size * num_dx + 4] - cX)
                    cX = numbers[rep_size * num_dx + 4]
                    out_list.append(numbers[rep_size * num_dx + 5] - cX)
                    cX = numbers[rep_size * num_dx + 5]
                    out_list.append(numbers[rep_size * num_dx + 6] - cY)
                    cY = numbers[rep_size * num_dx + 6]
                    out_list.append(numbers[rep_size * num_dx + 7] - cY)
                    cY = numbers[rep_size * num_dx + 7]
                if len(numbers) % rep_size == 1:
                    out_list.append(numbers[-1] - cX)
                    cX = numbers[-1]
            elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                out_list.append(numbers[0] - cY)
                cY = numbers[0]
                out_list.append(numbers[1] - cX)
                cX = numbers[1]
                out_list.append(numbers[2] - cY)
                cY = numbers[2]
                out_list.append(numbers[3] - cX)
                cX = numbers[3]
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx + 4] - cX)
                    cX = numbers[rep_size * num_dx + 4]
                    out_list.append(numbers[rep_size * num_dx + 5] - cX)
                    cX = numbers[rep_size * num_dx + 5]
                    out_list.append(numbers[rep_size * num_dx + 6] - cY)
                    cY = numbers[rep_size * num_dx + 6]
                    out_list.append(numbers[rep_size * num_dx + 7] - cY)
                    cY = numbers[rep_size * num_dx + 7]
                    out_list.append(numbers[rep_size * num_dx + 8] - cY)
                    cY = numbers[rep_size * num_dx + 8]
                    out_list.append(numbers[rep_size * num_dx + 9] - cX)
                    cX = numbers[rep_size * num_dx + 9]
                    out_list.append(numbers[rep_size * num_dx + 10] - cY)
                    cY = numbers[rep_size * num_dx + 10]
                    out_list.append(numbers[rep_size * num_dx + 11] - cX)
                    cX = numbers[rep_size * num_dx + 11]
                if len(numbers) % rep_size == 5:
                    out_list.append(numbers[-1] - cY)
                    cY = numbers[-1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                
        elif operator == "rcurveline":
            rep_size = 6
            if len(numbers) % rep_size == 2 and len(numbers) > 2:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(numbers[rep_size * num_dx] - cX)
                    cX = numbers[rep_size * num_dx]
                    out_list.append(numbers[rep_size * num_dx + 1] - cY)
                    cY = numbers[rep_size * num_dx + 1]
                    out_list.append(numbers[rep_size * num_dx + 2] - cX)
                    cX = numbers[rep_size * num_dx + 2]
                    out_list.append(numbers[rep_size * num_dx + 3] - cY)
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(numbers[rep_size * num_dx + 4] - cX)
                    cX = numbers[rep_size * num_dx + 4]
                    out_list.append(numbers[rep_size * num_dx + 5] - cY)
                    cY = numbers[rep_size * num_dx + 5]
                out_list.append(numbers[-2] - cX)
                cX = numbers[-2]
                out_list.append(numbers[-1] - cY)
                cY = numbers[-1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rlinecurve":
            rep_size = 2 # Repeat size
            if len(numbers) % rep_size == 0 and len(numbers) > 6:
                for num_dx in range((len(numbers) - 6) // rep_size):
                    # Extend path by two-dimension offset
                    out_list.append(numbers[rep_size * num_dx] - cX)
                    cX = numbers[rep_size * num_dx]
                    out_list.append(numbers[rep_size * num_dx + 1] - cY)
                    cY = numbers[rep_size * num_dx + 1]
                out_list.append(numbers[-6] - cX)
                cX = numbers[-6]
                out_list.append(numbers[-5] - cY)
                cY = numbers[-5]
                out_list.append(numbers[-4] - cX)
                cX = numbers[-4]
                out_list.append(numbers[-3] - cY)
                cY = numbers[-3]
                out_list.append(numbers[-2] - cX)
                cX = numbers[-2]
                out_list.append(numbers[-1] - cY)
                cY = numbers[-1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "flex":
            if len(numbers) == 13:
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
                out_list.append(numbers[1] - cY)
                cY = numbers[1]
                out_list.append(numbers[2] - cX)
                cX = numbers[2]
                out_list.append(numbers[3] - cY)
                cY = numbers[3]
                out_list.append(numbers[4] - cX)
                cX = numbers[4]
                out_list.append(numbers[5] - cY)
                cY = numbers[5]
                out_list.append(numbers[6] - cX)
                cX = numbers[6]
                out_list.append(numbers[7] - cY)
                cY = numbers[7]
                out_list.append(numbers[8] - cX)
                cX = numbers[8]
                out_list.append(numbers[9] - cY)
                cY = numbers[9]
                out_list.append(numbers[10] - cX)
                cX = numbers[10]
                out_list.append(numbers[11] - cY)
                cY = numbers[11]
                fd = numbers[12]
                out_list.append(fd)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hflex":
            if len(numbers) == 7:
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
                # cY += 0        # d1
                out_list.append(numbers[1] - cX)
                cX = numbers[1]
                out_list.append(numbers[2] - cY)
                cY = numbers[2] # d2
                out_list.append(numbers[3] - cX)
                cX = numbers[3]
                # cY += 0        # d3
                out_list.append(numbers[4] - cX)
                cX = numbers[4]
                # cY += 0        # d4
                out_list.append(numbers[5] - cX)
                cX = numbers[5]
                cY -= numbers[2] # d5 -- note this is reusing numbers[2] to return to same Y value
                # DO NOT ADD TO LIST ^^^
                out_list.append(numbers[6] - cX)
                cX = numbers[6]
                # cY += 0        # d6
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                
        elif operator == "hflex1":
            if len(numbers) == 9:
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
                out_list.append(numbers[1] - cY)
                cY = numbers[1] # d1
                out_list.append(numbers[2] - cX)
                cX = numbers[2]
                out_list.append(numbers[3] - cY)
                cY = numbers[3] # d2
                out_list.append(numbers[4] - cX)
                cX = numbers[4]
                # cY += 0        # d3
                out_list.append(numbers[5] - cX)
                cX = numbers[5]
                # cY += 0        # d4
                out_list.append(numbers[6] - cX)
                cX = numbers[6]
                out_list.append(numbers[7] - cY)
                cY = numbers[7] # d5
                out_list.append(numbers[8] - cX)
                cX = numbers[8]
                cY -= numbers[1] + numbers[3] + numbers[7] # d6 -- note this reuses numbers to return to same Y value
                # DO NOT ADD TO LIST ^^^
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "flex1":
            if len(numbers) == 11:
                # TODO: Support Bezier curves
                t_X = cX # temp X
                t_Y = cY # temp Y
                out_list.append(numbers[0] - cX)
                cX = numbers[0]
                out_list.append(numbers[1] - cY)
                cY = numbers[1] # d1
                out_list.append(numbers[2] - cX)
                cX = numbers[2]
                out_list.append(numbers[3] - cY)
                cY = numbers[3] # d2
                out_list.append(numbers[4] - cX)
                cX = numbers[4]
                out_list.append(numbers[5] - cY)
                cY = numbers[5]  # d3
                out_list.append(numbers[6] - cX)
                cX = numbers[6]
                out_list.append(numbers[7] - cY)
                cY = numbers[7] # d4
                out_list.append(numbers[8] - cX)
                cX = numbers[8]
                out_list.append(numbers[9] - cY)
                cY = numbers[9] # d5
                if abs(cX - t_X) > abs(cY - t_Y):
                    cX = numbers[10]
                    cY = t_Y
                else:
                    cX = t_X
                    cY = numbers[10]
                out_list.append(numbers[10])
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hstem":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "vstem":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "hstemhm":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "vstemhm":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "hintmask":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "cntrmask":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "callsubr":
            raise Exception("Operator not implemented")
        elif operator == "callgsubr":
            raise Exception("Operator not implemented")
        elif operator == "vsindex":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "blend":
            # Not in any of the fonts
            raise Exception("Operator not implemented")

        elif operator == "endchar":
            break

        else:
            raise Exception(f"Operator '{operator}' not found")

    if return_string:
        return [str(item) for item in out_list]
    else:
        return out_list
        

def center_and_scale(tablelist : list, tokenizer : Tokenizer, return_string : bool = True) -> list:
    '''
    Alters a default tablelist (operator first) such that it is centered
    around (0,0) and has a height of 300.

    Parameters:
    -----------
    tablelist (list[str]): the tablelist to alter (operators are first); must start with operator
    tokenizer (Tokenizer): the tokenizer
    return_string (bool): whether or not to return the numbers in string form

    Returns:
    --------
    list: the altered tablelist
    '''
    out_list = []

    tablelist, (cX_list, cY_list) = make_cumulative(operator_first(tablelist, tokenizer), tokenizer, return_string=False, return_stats=True)

    if len(tablelist) == 0 or tablelist[0] not in tokenizer.possible_operators:
        raise Exception("Tablelist must start with operator")

    min_X = min(cX_list)
    max_X = max(cX_list)
    mean_X = sum(cX_list) / len(cX_list)
    center_X = min_X + (max_X - min_X) // 2
    min_Y = min(cY_list)
    max_Y = max(cY_list)
    mean_Y = sum(cY_list) / len(cY_list)
    center_Y = min_Y + (max_Y - min_Y) // 2

    mid_X = round(mean_X)
    mid_Y = round(mean_Y)
    # scale_fac = min(max_X - min_X, max_Y - min_Y) / 300
    scale_fac = (max_Y - min_Y) / 300

    cX = 0
    cY = 0
    running_idx = 0
    while running_idx < len(tablelist):
        operator = tablelist[running_idx]
        out_list.append(operator)
        op_idx = running_idx
        running_idx += 1
        while running_idx < len(tablelist) and tablelist[running_idx] not in tokenizer.possible_operators \
            and tablelist[running_idx] != tokenizer.eos_token and tablelist[running_idx] != tokenizer.sos_token \
                and tablelist[running_idx] != tokenizer.pad_token:
            running_idx += 1

        numbers = [int(num) for num in tablelist[op_idx+1:running_idx]]
        
        if len(numbers) == 0 and operator != "endchar":
            continue

        if operator == "rmoveto":
            # First two numbers are coordinates; (optional) third is width
            if len(numbers) == 2:
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
                out_list.append(((numbers[1] - mid_Y) / scale_fac))
                cY = numbers[1]
            elif len(numbers) == 3 and running_idx == 4: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                out_list.append(((numbers[1] - mid_X) / scale_fac))
                cX = numbers[1]
                out_list.append(((numbers[2] - mid_Y) / scale_fac))
                cY = numbers[2]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "hmoveto":
            # First number is a coordinate; (optional) second is width
            if len(numbers) == 1:
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
            elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                out_list.append(((numbers[1] - mid_X) / scale_fac))
                cX = numbers[1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vmoveto":
            # First number is a coordinate; (optional) second is width
            if len(numbers) == 1:
                out_list.append(((numbers[0] - mid_Y) / scale_fac))
                cY = numbers[0]
            elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                width = numbers[0]
                out_list.append(width)
                out_list.append(((numbers[1] - mid_Y) / scale_fac))
                cY = numbers[1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rlineto":
            rep_size = 2 # Repeat size
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by two-dimension offset
                    out_list.append(((numbers[rep_size * num_dx] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx]
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "hlineto":
            rep_size = 2
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by x-dimension offset
                    out_list.append(((numbers[rep_size * num_dx] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx]
                    # Extend path by y-dimension offset
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 1]
            elif len(numbers) % rep_size == 1 and len(numbers) > 0:
                # Extend path by x-dimension offset
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by y-dimension offset
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 1]
                    # Extend path by x-dimension offset
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 2]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
        
        elif operator == "vlineto":
            rep_size = 2
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by y-dimension offset
                    out_list.append(((numbers[rep_size * num_dx] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx]
                    # Extend path by x-dimension offset
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 1]
            elif len(numbers) % rep_size == 1 and len(numbers) > 0:
                # Extend path by y-dimension offset
                out_list.append(((numbers[0] - mid_Y) / scale_fac))
                cY = numbers[0]
                for num_dx in range(len(numbers) // rep_size):
                    # Extend path by x-dimension offset
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 1]
                    # Extend path by y-dimension offset
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 2]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rrcurveto":
            rep_size = 6
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx]
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 1]
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 2]
                    out_list.append(((numbers[rep_size * num_dx + 3] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(((numbers[rep_size * num_dx + 4] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 4]
                    out_list.append(((numbers[rep_size * num_dx + 5] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 5]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hhcurveto":
            rep_size = 4
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx]
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 2]
                    out_list.append(((numbers[rep_size * num_dx + 3] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 3]
            elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                out_list.append(((numbers[0] - mid_Y) / scale_fac))
                cY = numbers[0]
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 2]
                    out_list.append(((numbers[rep_size * num_dx + 3] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(((numbers[rep_size * num_dx + 4] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 4]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vvcurveto":
            rep_size = 4
            if len(numbers) % rep_size == 0 and len(numbers) > 0:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx]
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 2]
                    out_list.append(((numbers[rep_size * num_dx + 3] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 3]
            elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 1]
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 2]
                    out_list.append(((numbers[rep_size * num_dx + 3] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(((numbers[rep_size * num_dx + 4] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 4]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hvcurveto":
            rep_size = 8
            if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx]
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 2]
                    out_list.append(((numbers[rep_size * num_dx + 3] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(((numbers[rep_size * num_dx + 4] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 4]
                    out_list.append(((numbers[rep_size * num_dx + 5] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 5]
                    out_list.append(((numbers[rep_size * num_dx + 6] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 6]
                    out_list.append(((numbers[rep_size * num_dx + 7] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 7]
                if len(numbers) % rep_size == 1:
                    out_list.append(((numbers[-1] - mid_Y) / scale_fac))
                    cY = numbers[-1]
            elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
                out_list.append(((numbers[1] - mid_X) / scale_fac))
                cX = numbers[1]
                out_list.append(((numbers[2] - mid_Y) / scale_fac))
                cY = numbers[2]
                out_list.append(((numbers[3] - mid_Y) / scale_fac))
                cY = numbers[3]
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx + 4] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 4]
                    out_list.append(((numbers[rep_size * num_dx + 5] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 5]
                    out_list.append(((numbers[rep_size * num_dx + 6] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 6]
                    out_list.append(((numbers[rep_size * num_dx + 7] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 7]
                    out_list.append(((numbers[rep_size * num_dx + 8] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 8]
                    out_list.append(((numbers[rep_size * num_dx + 9] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 9]
                    out_list.append(((numbers[rep_size * num_dx + 10] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 10]
                    out_list.append(((numbers[rep_size * num_dx + 11] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 11]
                if len(numbers) % rep_size == 5:
                    out_list.append(((numbers[-1] - mid_X) / scale_fac))
                    cX = numbers[-1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "vhcurveto":
            rep_size = 8
            if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx]
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 1]
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 2]
                    out_list.append(((numbers[rep_size * num_dx + 3] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 3]
                    out_list.append(((numbers[rep_size * num_dx + 4] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 4]
                    out_list.append(((numbers[rep_size * num_dx + 5] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 5]
                    out_list.append(((numbers[rep_size * num_dx + 6] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 6]
                    out_list.append(((numbers[rep_size * num_dx + 7] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 7]
                if len(numbers) % rep_size == 1:
                    out_list.append(((numbers[-1] - mid_X) / scale_fac))
                    cX = numbers[-1]
            elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                out_list.append(((numbers[0] - mid_Y) / scale_fac))
                cY = numbers[0]
                out_list.append(((numbers[1] - mid_X) / scale_fac))
                cX = numbers[1]
                out_list.append(((numbers[2] - mid_Y) / scale_fac))
                cY = numbers[2]
                out_list.append(((numbers[3] - mid_X) / scale_fac))
                cX = numbers[3]
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx + 4] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 4]
                    out_list.append(((numbers[rep_size * num_dx + 5] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 5]
                    out_list.append(((numbers[rep_size * num_dx + 6] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 6]
                    out_list.append(((numbers[rep_size * num_dx + 7] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 7]
                    out_list.append(((numbers[rep_size * num_dx + 8] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 8]
                    out_list.append(((numbers[rep_size * num_dx + 9] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 9]
                    out_list.append(((numbers[rep_size * num_dx + 10] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 10]
                    out_list.append(((numbers[rep_size * num_dx + 11] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 11]
                if len(numbers) % rep_size == 5:
                    out_list.append(((numbers[-1] - mid_Y) / scale_fac))
                    cY = numbers[-1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                
        elif operator == "rcurveline":
            rep_size = 6
            if len(numbers) % rep_size == 2 and len(numbers) > 2:
                for num_dx in range(len(numbers) // rep_size):
                    out_list.append(((numbers[rep_size * num_dx] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx]
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 1]
                    out_list.append(((numbers[rep_size * num_dx + 2] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 2]
                    out_list.append(((numbers[rep_size * num_dx + 3] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 3]
                    out_list.append(((numbers[rep_size * num_dx + 4] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx + 4]
                    out_list.append(((numbers[rep_size * num_dx + 5] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 5]
                out_list.append(((numbers[-2] - mid_X) / scale_fac))
                cX = numbers[-2]
                out_list.append(((numbers[-1] - mid_Y) / scale_fac))
                cY = numbers[-1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "rlinecurve":
            rep_size = 2 # Repeat size
            if len(numbers) % rep_size == 0 and len(numbers) > 6:
                for num_dx in range((len(numbers) - 6) // rep_size):
                    # Extend path by two-dimension offset
                    out_list.append(((numbers[rep_size * num_dx] - mid_X) / scale_fac))
                    cX = numbers[rep_size * num_dx]
                    out_list.append(((numbers[rep_size * num_dx + 1] - mid_Y) / scale_fac))
                    cY = numbers[rep_size * num_dx + 1]
                out_list.append(((numbers[-6] - mid_X) / scale_fac))
                cX = numbers[-6]
                out_list.append(((numbers[-5] - mid_Y) / scale_fac))
                cY = numbers[-5]
                out_list.append(((numbers[-4] - mid_X) / scale_fac))
                cX = numbers[-4]
                out_list.append(((numbers[-3] - mid_Y) / scale_fac))
                cY = numbers[-3]
                out_list.append(((numbers[-2] - mid_X) / scale_fac))
                cX = numbers[-2]
                out_list.append(((numbers[-1] - mid_Y) / scale_fac))
                cY = numbers[-1]
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "flex":
            if len(numbers) == 13:
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
                out_list.append(((numbers[1] - mid_Y) / scale_fac))
                cY = numbers[1]
                out_list.append(((numbers[2] - mid_X) / scale_fac))
                cX = numbers[2]
                out_list.append(((numbers[3] - mid_Y) / scale_fac))
                cY = numbers[3]
                out_list.append(((numbers[4] - mid_X) / scale_fac))
                cX = numbers[4]
                out_list.append(((numbers[5] - mid_Y) / scale_fac))
                cY = numbers[5]
                out_list.append(((numbers[6] - mid_X) / scale_fac))
                cX = numbers[6]
                out_list.append(((numbers[7] - mid_Y) / scale_fac))
                cY = numbers[7]
                out_list.append(((numbers[8] - mid_X) / scale_fac))
                cX = numbers[8]
                out_list.append(((numbers[9] - mid_Y) / scale_fac))
                cY = numbers[9]
                out_list.append(((numbers[10] - mid_X) / scale_fac))
                cX = numbers[10]
                out_list.append(((numbers[11] - mid_Y) / scale_fac))
                cY = numbers[11]
                fd = numbers[12]
                out_list.append(fd)
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hflex":
            if len(numbers) == 7:
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
                # cY += 0        # d1
                out_list.append(((numbers[1] - mid_X) / scale_fac))
                cX = numbers[1]
                out_list.append(((numbers[2] - mid_Y) / scale_fac))
                cY = numbers[2] # d2
                out_list.append(((numbers[3] - mid_X) / scale_fac))
                cX = numbers[3]
                # cY += 0        # d3
                out_list.append(((numbers[4] - mid_X) / scale_fac))
                cX = numbers[4]
                # cY += 0        # d4
                out_list.append(((numbers[5] - mid_X) / scale_fac))
                cX = numbers[5]
                cY -= numbers[2] # d5 -- note this is reusing numbers[2] to return to same Y value
                # DO NOT ADD TO LIST ^^^
                out_list.append(((numbers[6] - mid_X) / scale_fac))
                cX = numbers[6]
                # cY += 0        # d6
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                
        elif operator == "hflex1":
            if len(numbers) == 9:
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
                out_list.append(((numbers[1] - mid_Y) / scale_fac))
                cY = numbers[1] # d1
                out_list.append(((numbers[2] - mid_X) / scale_fac))
                cX = numbers[2]
                out_list.append(((numbers[3] - mid_Y) / scale_fac))
                cY = numbers[3] # d2
                out_list.append(((numbers[4] - mid_X) / scale_fac))
                cX = numbers[4]
                # cY += 0        # d3
                out_list.append(((numbers[5] - mid_X) / scale_fac))
                cX = numbers[5]
                # cY += 0        # d4
                out_list.append(((numbers[6] - mid_X) / scale_fac))
                cX = numbers[6]
                out_list.append(((numbers[7] - mid_Y) / scale_fac))
                cY = numbers[7] # d5
                out_list.append(((numbers[8] - mid_X) / scale_fac))
                cX = numbers[8]
                cY -= numbers[1] + numbers[3] + numbers[7] # d6 -- note this reuses numbers to return to same Y value
                # DO NOT ADD TO LIST ^^^
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "flex1":
            if len(numbers) == 11:
                # TODO: Support Bezier curves
                t_X = cX # temp X
                t_Y = cY # temp Y
                out_list.append(((numbers[0] - mid_X) / scale_fac))
                cX = numbers[0]
                out_list.append(((numbers[1] - mid_Y) / scale_fac))
                cY = numbers[1] # d1
                out_list.append(((numbers[2] - mid_X) / scale_fac))
                cX = numbers[2]
                out_list.append(((numbers[3] - mid_Y) / scale_fac))
                cY = numbers[3] # d2
                out_list.append(((numbers[4] - mid_X) / scale_fac))
                cX = numbers[4]
                out_list.append(((numbers[5] - mid_Y) / scale_fac))
                cY = numbers[5]  # d3
                out_list.append(((numbers[6] - mid_X) / scale_fac))
                cX = numbers[6]
                out_list.append(((numbers[7] - mid_Y) / scale_fac))
                cY = numbers[7] # d4
                out_list.append(((numbers[8] - mid_X) / scale_fac))
                cX = numbers[8]
                out_list.append(((numbers[9] - mid_Y) / scale_fac))
                cY = numbers[9] # d5
                if abs(cX - t_X) > abs(cY - t_Y):
                    cX = numbers[10]
                    cY = t_Y
                    out_list.append(((numbers[10] - mid_X) / scale_fac))
                else:
                    cX = t_X
                    cY = numbers[10]
                    out_list.append(((numbers[10] - mid_Y) / scale_fac))
                fd = 50
            else:
                raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

        elif operator == "hstem":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "vstem":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "hstemhm":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "vstemhm":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "hintmask":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "cntrmask":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "callsubr":
            raise Exception("Operator not implemented")
        elif operator == "callgsubr":
            raise Exception("Operator not implemented")
        elif operator == "vsindex":
            # Not in any of the fonts
            raise Exception("Operator not implemented")
        elif operator == "blend":
            # Not in any of the fonts
            raise Exception("Operator not implemented")

        elif operator == "endchar":
            break

        else:
            raise Exception(f"Operator '{operator}' not found")

    return numbers_first(make_non_cumulative(out_list, tokenizer), tokenizer, return_string=return_string)


if __name__ == "__main__":
    min_number = -1000
    max_number = 1000
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

    tablelist = ['5', '3', '1', "rmoveto", '5', '3', '1', '-1', '0', '1', "rrcurveto", '-1', '-1', "hlineto", "endchar"]
    inverted_tablelist = ["rmoveto", '5', '3', '1', "rrcurveto", '5', '3', '1', '-1', '0', '1', "hlineto", '-1', '-1', "endchar"]
    cumulative_tablelist = ["rmoveto", '5', '3', '1', "rrcurveto", '8', '4', '9', '3', '9', '4', "hlineto", '8', '3', "endchar"]

    op_first = operator_first(tablelist, tokenizer)
    op_last = numbers_first(inverted_tablelist, tokenizer)
    cumulative = make_cumulative(inverted_tablelist, tokenizer)
    non_cumulative = make_non_cumulative(cumulative_tablelist, tokenizer)

    assert inverted_tablelist == op_first, "FAIL: 1"
    assert tablelist == op_last, "FAIL: 2"
    assert cumulative_tablelist == cumulative, "FAIL: 3"
    assert inverted_tablelist == non_cumulative, "FAIL: 4"

    print("Passed")
