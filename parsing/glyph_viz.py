import numpy as np
import matplotlib.pyplot as plt
from config import operators


class Paths:
    def __init__(self):
        self.paths = []

    def new_path(self, cX, cY):
        self.paths.append([(cX, cY)])

    def extend(self, cX, cY):
        self.paths[-1].append((cX, cY))

    def get_paths(self):
        return self.paths


class Visualizer:
    def __init__(self, table_list : list):
        self.table_list = table_list

    def get_next_operator(self, index) -> tuple[int, list, str]:
        '''
        Given an index of the table list, finds the next non-operator index and returns that index,
        as well as the list of numbers used by the operator, as well as the operator itself
        '''
        next_index = index
        while next_index < len(self.table_list) and isinstance(self.table_list[next_index], str) is False:
            next_index += 1
        return next_index + 1, self.table_list[index:next_index], self.table_list[next_index]

    def get_paths(self) -> list[list[tuple]]:
        '''
        Returns the paths of each of the contours, in the format: [[(path1_idx0_x, path1_idx0_y), ...], ...]
        '''
        running_idx = 0
        paths = Paths()
        cX, cY = 0
        width = None
        while running_idx < len(self.table_list):
            running_idx, numbers, operator = self.get_next_operator(running_idx)

            if operator == "rmoveto":
                # First two numbers are coordinates; (optional) third is width
                if len(numbers) == 2:
                    cX += numbers[0]
                    cY += numbers[1]
                    paths.new_path(cX, cY)
                elif len(numbers) == 3 and running_idx == 4: # must be first operator in sequence
                    cX += numbers[0]
                    cY += numbers[1]
                    paths.new_path(cX, cY)
                    width = numbers[2]
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
            
            elif operator == "hmoveto":
                # First number is a coordinate; (optional) second is width
                if len(numbers) == 1:
                    cX += numbers[0]
                    paths.new_path(cX, cY)
                elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                    cX += numbers[0]
                    paths.new_path(cX, cY)
                    width = numbers[2]
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "vmoveto":
                # First number is a coordinate; (optional) second is width
                if len(numbers) == 1:
                    cY += numbers[0]
                    paths.new_path(cX, cY)
                elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                    cY += numbers[0]
                    paths.new_path(cX, cY)
                    width = numbers[2]
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "rlineto":
                rep_size = 2 # Repeat size
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by two-dimension offset
                        cX += numbers[rep_size * num_dx]
                        cY += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
            
            elif operator == "hlineto":
                rep_size = 2
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by x-dimension offset
                        cX += numbers[rep_size * num_dx]
                        paths.extend(cX, cY)
                        # Extend path by y-dimension offset
                        cY += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                if len(numbers) % rep_size == 1 and len(numbers) > 0:
                    # Extend path by x-dimension offset
                    cX += numbers[0]
                    paths.extend(cX, cY)
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by y-dimension offset
                        cY += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                        # Extend path by x-dimension offset
                        cX += numbers[rep_size * num_dx + 2]
                        paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
            
            elif operator == "vlineto":
                rep_size = 2
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by y-dimension offset
                        cY += numbers[rep_size * num_dx]
                        paths.extend(cX, cY)
                        # Extend path by x-dimension offset
                        cX += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                if len(numbers) % rep_size == 1 and len(numbers) > 0:
                    # Extend path by y-dimension offset
                    cY += numbers[0]
                    paths.extend(cX, cY)
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by x-dimension offset
                        cX += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                        # Extend path by y-dimension offset
                        cY += numbers[rep_size * num_dx + 2]
                        paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "rrcurveto":
                # TODO: Support Bezier curves
                rep_size = 6
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cX += numbers[rep_size * num_dx]
                        cY += numbers[rep_size * num_dx + 1]
                        cX += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        cX += numbers[rep_size * num_dx + 4]
                        cY += numbers[rep_size * num_dx + 5]
                        paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "hhcurveto":
                # TODO: Support Bezier curves
                rep_size = 4
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cX += numbers[rep_size * num_dx]
                        cX += numbers[rep_size * num_dx + 1]
                        cY += numbers[rep_size * num_dx + 2]
                        cX += numbers[rep_size * num_dx + 3]
                        paths.extend(cX, cY)
                elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                    cY += numbers[0]
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cX += numbers[rep_size * num_dx + 1]
                        cX += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        cX += numbers[rep_size * num_dx + 4]
                        paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "vvcurveto":
                # TODO: Support Bezier curves
                rep_size = 4
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cY += numbers[rep_size * num_dx]
                        cX += numbers[rep_size * num_dx + 1]
                        cY += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        paths.extend(cX, cY)
                elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                    cX += numbers[0]
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cY += numbers[rep_size * num_dx + 1]
                        cX += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        cY += numbers[rep_size * num_dx + 4]
                        paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "hvcurveto":
                # TODO: Support Bezier curves
                rep_size = 8
                if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cX += numbers[rep_size * num_dx]
                        cX += numbers[rep_size * num_dx + 1]
                        cY += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        paths.extend(cX, cY) # A reasonable middle point in the Bezier
                        cY += numbers[rep_size * num_dx + 4]
                        cX += numbers[rep_size * num_dx + 5]
                        cY += numbers[rep_size * num_dx + 6]
                        cX += numbers[rep_size * num_dx + 7]
                        paths.extend(cX, cY)
                    if len(numbers) % rep_size == 1:
                        cY += numbers[-1]
                        paths.extend(cX, cY)
                elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                    cX += numbers[0]
                    cX += numbers[1]
                    cY += numbers[2]
                    cY += numbers[3]
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cY += numbers[rep_size * num_dx + 4]
                        cX += numbers[rep_size * num_dx + 5]
                        cY += numbers[rep_size * num_dx + 6]
                        cX += numbers[rep_size * num_dx + 7]
                        paths.extend(cX, cY) # A reasonable middle point in the Bezier
                        cX += numbers[rep_size * num_dx + 8]
                        cX += numbers[rep_size * num_dx + 9]
                        cY += numbers[rep_size * num_dx + 10]
                        cY += numbers[rep_size * num_dx + 11]
                        paths.extend(cX, cY)
                    if len(numbers) % rep_size == 5:
                        cX += numbers[-1]
                        paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "vhcurveto":
                # TODO: Support Bezier curves
                rep_size = 8
                if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cY += numbers[rep_size * num_dx]
                        cX += numbers[rep_size * num_dx + 1]
                        cY += numbers[rep_size * num_dx + 2]
                        cX += numbers[rep_size * num_dx + 3]
                        paths.extend(cX, cY) # A reasonable middle point in the Bezier
                        cX += numbers[rep_size * num_dx + 4]
                        cX += numbers[rep_size * num_dx + 5]
                        cY += numbers[rep_size * num_dx + 6]
                        cY += numbers[rep_size * num_dx + 7]
                        paths.extend(cX, cY)
                    if len(numbers) % rep_size == 1:
                        cX += numbers[-1]
                        paths.extend(cX, cY)
                elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                    cY += numbers[0]
                    cX += numbers[1]
                    cY += numbers[2]
                    cX += numbers[3]
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cX += numbers[rep_size * num_dx + 4]
                        cX += numbers[rep_size * num_dx + 5]
                        cY += numbers[rep_size * num_dx + 6]
                        cY += numbers[rep_size * num_dx + 7]
                        paths.extend(cX, cY) # A reasonable middle point in the Bezier
                        cY += numbers[rep_size * num_dx + 8]
                        cX += numbers[rep_size * num_dx + 9]
                        cY += numbers[rep_size * num_dx + 10]
                        cX += numbers[rep_size * num_dx + 11]
                        paths.extend(cX, cY)
                    if len(numbers) % rep_size == 5:
                        cY += numbers[-1]
                        paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                    
            elif operator == "rcurveline":
                # TODO: Support Bezier curves
                rep_size = 6
                if len(numbers) % rep_size == 2 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # TODO: Support Bezier curves
                        cX += numbers[rep_size * num_dx]
                        cY += numbers[rep_size * num_dx + 1]
                        cX += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        cX += numbers[rep_size * num_dx + 4]
                        cY += numbers[rep_size * num_dx + 5]
                        paths.extend(cX, cY)
                    cX += numbers[-2]
                    cY += numbers[-1]
                    paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "rlinecurve":
                rep_size = 2 # Repeat size
                if len(numbers) % rep_size == 0 and len(numbers) > 6:
                    for num_dx in range((len(numbers) - 6) // rep_size):
                        # Extend path by two-dimension offset
                        cX += numbers[rep_size * num_dx]
                        cY += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                    # TODO: Support Bezier curves
                    cX += numbers[-6]
                    cY += numbers[-5]
                    cX += numbers[-4]
                    cY += numbers[-3]
                    cX += numbers[-2]
                    cY += numbers[-1]
                    paths.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "flex":
                # TODO: Support Bezier curves
                if len(numbers) == 13:
                    # TODO: Support Bezier curves
                    cX += numbers[0]
                    cY += numbers[1]
                    cX += numbers[2]
                    cY += numbers[3]
                    cX += numbers[4]
                    cY += numbers[5]
                    paths.extend(cX, cY)
                    cX += numbers[6]
                    cY += numbers[7]
                    cX += numbers[8]
                    cY += numbers[9]
                    cX += numbers[10]
                    cY += numbers[11]
                    paths.extend(cX, cY)
                    fd = numbers[12]
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "hflex":
                # TODO: Support Bezier curves
                if len(numbers) == 7:
                    # TODO: Support Bezier curves
                    cX += numbers[0]
                    # cY += 0        # d1
                    cX += numbers[1]
                    cY += numbers[2] # d2
                    cX += numbers[3]
                    # cY += 0        # d3
                    paths.extend(cX, cY)
                    cX += numbers[4]
                    # cY += 0        # d4
                    cX += numbers[5]
                    cY -= numbers[2] # d5 -- note this is reusing numbers[2] to return to same Y value
                    cX += numbers[6]
                    # cY += 0        # d6
                    paths.extend(cX, cY)
                    fd = 50
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                    
            elif operator == "hflex1":
                # TODO: Support Bezier curves
                if len(numbers) == 9:
                    # TODO: Support Bezier curves
                    cX += numbers[0]
                    cY += numbers[1] # d1
                    cX += numbers[2]
                    cY += numbers[3] # d2
                    cX += numbers[4]
                    # cY += 0        # d3
                    paths.extend(cX, cY)
                    cX += numbers[5]
                    # cY += 0        # d4
                    cX += numbers[6]
                    cY += numbers[7] # d5
                    cX += numbers[8]
                    cY -= numbers[1] + numbers[3] + numbers[7] # d6 -- note this reuses numbers to return to same Y value
                    paths.extend(cX, cY)
                    fd = 50
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "flex1":
                # TODO: Support Bezier curves
                if len(numbers) == 11:
                    # TODO: Support Bezier curves
                    t_X = cX # temp X
                    t_Y = cY # temp Y
                    cX += numbers[0]
                    cY += numbers[1] # d1
                    cX += numbers[2]
                    cY += numbers[3] # d2
                    cX += numbers[4]
                    cY += numbers[5]  # d3
                    paths.extend(cX, cY)
                    cX += numbers[6]
                    cY += numbers[7] # d4
                    cX += numbers[8]
                    cY += numbers[9] # d5
                    if abs(cX - t_X) > abs(cY - t_Y):
                        cX = numbers[10]
                        cY = t_Y
                    else:
                        cX = t_X
                        cY = numbers[10]
                    paths.extend(cX, cY)
                    fd = 50
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "hstem":
                pass
            elif operator == "vstem":
                pass
            elif operator == "hstemhm":
                pass
            elif operator == "vstemhm":
                pass
            elif operator == "hintmask":
                pass
            elif operator == "cntrmask":
                pass
            elif operator == "callsubr":
                pass
            elif operator == "callgsubr":
                pass
            elif operator == "vsindex":
                pass
            elif operator == "blend":
                pass
            
            elif operator == "endchar":
                return
            
            
        return paths

    def draw(self):
        paths = self.get_paths()
        # TODO: matplotlib stuff