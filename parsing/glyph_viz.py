import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mp
from matplotlib.patches import PathPatch


def cubic_bezier(a1, c1, c2, a2, inter = 25):
    points = [a1]
    for idx in range(inter):
        prc = (idx + 1) / (inter + 1)
        m1 = (a1[0] + (c1[0] - a1[0]) * prc, a1[1] + (c1[1] - a1[1]) * prc)
        m2 = (c2[0] + (a2[0] - c2[0]) * prc, c2[1] + (a2[1] - c2[1]) * prc)
        m3 = (m1[0] + (m2[0] - m1[0]) * prc, m1[1] + (m2[1] - m1[1]) * prc)
        points.append(m3)
    points.append(a2)
    return points


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
        as well as the list of numbers used by the operator, as well as the operator itself. If the
        end of the table list is reached without an operator at the end, a None is returned for the
        operator.
        '''
        next_index = index
        while next_index < len(self.table_list) and isinstance(self.table_list[next_index], str) is False:
            next_index += 1
        
        if next_index == len(self.table_list):
            return next_index, self.table_list[index:next_index], None
        else:
            return next_index + 1, self.table_list[index:next_index], self.table_list[next_index]

    def get_paths(self) -> list[list[tuple]]:
        '''
        Returns the paths of each of the contours, in the format: [[(path1_idx0_x, path1_idx0_y), ...], ...]
        '''
        running_idx = 0
        paths = Paths()
        control_points = Paths()
        cX = 0
        cY = 0
        width = None
        while running_idx < len(self.table_list):
            running_idx, numbers, operator = self.get_next_operator(running_idx)

            if operator == "rmoveto":
                # First two numbers are coordinates; (optional) third is width
                if len(numbers) == 2:
                    cX += numbers[0]
                    cY += numbers[1]
                    paths.new_path(cX, cY)
                    control_points.new_path(cX, cY)
                elif len(numbers) == 3 and running_idx == 4: # must be first operator in sequence
                    width = numbers[0]
                    cX += numbers[1]
                    cY += numbers[2]
                    paths.new_path(cX, cY)
                    control_points.new_path(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
            
            elif operator == "hmoveto":
                # First number is a coordinate; (optional) second is width
                if len(numbers) == 1:
                    cX += numbers[0]
                    paths.new_path(cX, cY)
                    control_points.new_path(cX, cY)
                elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                    width = numbers[0]
                    cX += numbers[1]
                    paths.new_path(cX, cY)
                    control_points.new_path(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "vmoveto":
                # First number is a coordinate; (optional) second is width
                if len(numbers) == 1:
                    cY += numbers[0]
                    paths.new_path(cX, cY)
                    control_points.new_path(cX, cY)
                elif len(numbers) == 2 and running_idx == 3: # must be first operator in sequence
                    width = numbers[0]
                    cY += numbers[1]
                    paths.new_path(cX, cY)
                    control_points.new_path(cX, cY)
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
                        control_points.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
            
            elif operator == "hlineto":
                rep_size = 2
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by x-dimension offset
                        cX += numbers[rep_size * num_dx]
                        paths.extend(cX, cY)
                        control_points.extend(cX, cY)
                        # Extend path by y-dimension offset
                        cY += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                        control_points.extend(cX, cY)
                elif len(numbers) % rep_size == 1 and len(numbers) > 0:
                    # Extend path by x-dimension offset
                    cX += numbers[0]
                    paths.extend(cX, cY)
                    control_points.extend(cX, cY)
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by y-dimension offset
                        cY += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                        control_points.extend(cX, cY)
                        # Extend path by x-dimension offset
                        cX += numbers[rep_size * num_dx + 2]
                        paths.extend(cX, cY)
                        control_points.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
            
            elif operator == "vlineto":
                rep_size = 2
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by y-dimension offset
                        cY += numbers[rep_size * num_dx]
                        paths.extend(cX, cY)
                        control_points.extend(cX, cY)
                        # Extend path by x-dimension offset
                        cX += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                        control_points.extend(cX, cY)
                elif len(numbers) % rep_size == 1 and len(numbers) > 0:
                    # Extend path by y-dimension offset
                    cY += numbers[0]
                    paths.extend(cX, cY)
                    control_points.extend(cX, cY)
                    for num_dx in range(len(numbers) // rep_size):
                        # Extend path by x-dimension offset
                        cX += numbers[rep_size * num_dx + 1]
                        paths.extend(cX, cY)
                        control_points.extend(cX, cY)
                        # Extend path by y-dimension offset
                        cY += numbers[rep_size * num_dx + 2]
                        paths.extend(cX, cY)
                        control_points.extend(cX, cY)
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "rrcurveto":
                rep_size = 6
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        a1 = (cX, cY)
                        cX += numbers[rep_size * num_dx]
                        cY += numbers[rep_size * num_dx + 1]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        c2 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 4]
                        cY += numbers[rep_size * num_dx + 5]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "hhcurveto":
                rep_size = 4
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        a1 = (cX, cY)
                        cX += numbers[rep_size * num_dx]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 1]
                        cY += numbers[rep_size * num_dx + 2]
                        c2 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 3]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                    a1 = (cX, cY)
                    cY += numbers[0]
                    for num_dx in range(len(numbers) // rep_size):
                        if num_dx != 0:
                            a1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 1]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        c2 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 4]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "vvcurveto":
                rep_size = 4
                if len(numbers) % rep_size == 0 and len(numbers) > 0:
                    for num_dx in range(len(numbers) // rep_size):
                        a1 = (cX, cY)
                        cY += numbers[rep_size * num_dx]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 1]
                        cY += numbers[rep_size * num_dx + 2]
                        c2 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 3]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                elif len(numbers) % rep_size == 1 and len(numbers) > 1:
                    a1 = (cX, cY)
                    cX += numbers[0]
                    for num_dx in range(len(numbers) // rep_size):
                        if num_dx != 0:
                            a1 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 1]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        c2 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 4]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "hvcurveto":
                rep_size = 8
                if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                    for num_dx in range(len(numbers) // rep_size):
                        a1 = (cX, cY)
                        cX += numbers[rep_size * num_dx]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 1]
                        cY += numbers[rep_size * num_dx + 2]
                        c2 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 3]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                        a1 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 4]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 5]
                        cY += numbers[rep_size * num_dx + 6]
                        c2 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 7]
                        a2 = (cX, cY)
                        if num_dx + 1 < len(numbers) // rep_size or len(numbers) % rep_size != 1:
                            control_points.extend(cX, cY)
                            for pt in cubic_bezier(a1, c1, c2, a2):
                                paths.extend(pt[0], pt[1])
                    if len(numbers) % rep_size == 1:
                        cY += numbers[-1]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                    a1 = (cX, cY)
                    cX += numbers[0]
                    c1 = (cX, cY)
                    cX += numbers[1]
                    cY += numbers[2]
                    c2 = (cX, cY)
                    cY += numbers[3]
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    for num_dx in range(len(numbers) // rep_size):
                        a1 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 4]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 5]
                        cY += numbers[rep_size * num_dx + 6]
                        c2 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 7]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                        a1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 8]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 9]
                        cY += numbers[rep_size * num_dx + 10]
                        c2 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 11]
                        a2 = (cX, cY)
                        if num_dx + 1 < len(numbers) // rep_size or len(numbers) % rep_size != 5:
                            control_points.extend(cX, cY)
                            for pt in cubic_bezier(a1, c1, c2, a2):
                                paths.extend(pt[0], pt[1])
                    if len(numbers) % rep_size == 5:
                        cX += numbers[-1]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "vhcurveto":
                rep_size = 8
                if (len(numbers) % rep_size == 0 or len(numbers) % rep_size == 1) and len(numbers) > 1:
                    for num_dx in range(len(numbers) // rep_size):
                        a1 = (cX, cY)
                        cY += numbers[rep_size * num_dx]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 1]
                        cY += numbers[rep_size * num_dx + 2]
                        c2 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 3]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                        a1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 4]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 5]
                        cY += numbers[rep_size * num_dx + 6]
                        c2 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 7]
                        a2 = (cX, cY)
                        if num_dx + 1 < len(numbers) // rep_size or len(numbers) % rep_size != 1:
                            control_points.extend(cX, cY)
                            for pt in cubic_bezier(a1, c1, c2, a2):
                                paths.extend(pt[0], pt[1])
                    if len(numbers) % rep_size == 1:
                        cX += numbers[-1]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                elif (len(numbers) % rep_size == 4 or len(numbers) % rep_size == 5) and len(numbers) > 0:
                    a1 = (cX, cY)
                    cY += numbers[0]
                    c1 = (cX, cY)
                    cX += numbers[1]
                    cY += numbers[2]
                    c2 = (cX, cY)
                    cX += numbers[3]
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    for num_dx in range(len(numbers) // rep_size):
                        a1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 4]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 5]
                        cY += numbers[rep_size * num_dx + 6]
                        c2 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 7]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                        a1 = (cX, cY)
                        cY += numbers[rep_size * num_dx + 8]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 9]
                        cY += numbers[rep_size * num_dx + 10]
                        c2 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 11]
                        a2 = (cX, cY)
                        if num_dx + 1 < len(numbers) // rep_size or len(numbers) % rep_size != 5:
                            control_points.extend(cX, cY)
                            for pt in cubic_bezier(a1, c1, c2, a2):
                                paths.extend(pt[0], pt[1])
                    if len(numbers) % rep_size == 5:
                        cY += numbers[-1]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                    
            elif operator == "rcurveline":
                rep_size = 6
                if len(numbers) % rep_size == 2 and len(numbers) > 2:
                    for num_dx in range(len(numbers) // rep_size):
                        a1 = (cX, cY)
                        cX += numbers[rep_size * num_dx]
                        cY += numbers[rep_size * num_dx + 1]
                        c1 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 2]
                        cY += numbers[rep_size * num_dx + 3]
                        c2 = (cX, cY)
                        cX += numbers[rep_size * num_dx + 4]
                        cY += numbers[rep_size * num_dx + 5]
                        a2 = (cX, cY)
                        control_points.extend(cX, cY)
                        for pt in cubic_bezier(a1, c1, c2, a2):
                            paths.extend(pt[0], pt[1])
                    cX += numbers[-2]
                    cY += numbers[-1]
                    paths.extend(cX, cY)
                    control_points.extend(cX, cY)
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
                        control_points.extend(cX, cY)
                    a1 = (cX, cY)
                    cX += numbers[-6]
                    cY += numbers[-5]
                    c1 = (cX, cY)
                    cX += numbers[-4]
                    cY += numbers[-3]
                    c2 = (cX, cY)
                    cX += numbers[-2]
                    cY += numbers[-1]
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "flex":
                if len(numbers) == 13:
                    a1 = (cX, cY)
                    cX += numbers[0]
                    cY += numbers[1]
                    c1 = (cX, cY)
                    cX += numbers[2]
                    cY += numbers[3]
                    c2 = (cX, cY)
                    cX += numbers[4]
                    cY += numbers[5]
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    a1 = (cX, cY)
                    cX += numbers[6]
                    cY += numbers[7]
                    c1 = (cX, cY)
                    cX += numbers[8]
                    cY += numbers[9]
                    c2 = (cX, cY)
                    cX += numbers[10]
                    cY += numbers[11]
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    fd = numbers[12]
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "hflex":
                if len(numbers) == 7:
                    a1 = (cX, cY)
                    cX += numbers[0]
                    # cY += 0        # d1
                    c1 = (cX, cY)
                    cX += numbers[1]
                    cY += numbers[2] # d2
                    c2 = (cX, cY)
                    cX += numbers[3]
                    # cY += 0        # d3
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    a1 = (cX, cY)
                    cX += numbers[4]
                    # cY += 0        # d4
                    c1 = (cX, cY)
                    cX += numbers[5]
                    cY -= numbers[2] # d5 -- note this is reusing numbers[2] to return to same Y value
                    c2 = (cX, cY)
                    cX += numbers[6]
                    # cY += 0        # d6
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    fd = 50
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")
                    
            elif operator == "hflex1":
                if len(numbers) == 9:
                    a1 = (cX, cY)
                    cX += numbers[0]
                    cY += numbers[1] # d1
                    c1 = (cX, cY)
                    cX += numbers[2]
                    cY += numbers[3] # d2
                    c2 = (cX, cY)
                    cX += numbers[4]
                    # cY += 0        # d3
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    a1 = (cX, cY)
                    cX += numbers[5]
                    # cY += 0        # d4
                    c1 = (cX, cY)
                    cX += numbers[6]
                    cY += numbers[7] # d5
                    c2 = (cX, cY)
                    cX += numbers[8]
                    cY -= numbers[1] + numbers[3] + numbers[7] # d6 -- note this reuses numbers to return to same Y value
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    fd = 50
                else:
                    raise Exception(f"{operator} at index {running_idx - 1} has wrong coordinate count ({len(numbers)})")

            elif operator == "flex1":
                if len(numbers) == 11:
                    # TODO: Support Bezier curves
                    t_X = cX # temp X
                    t_Y = cY # temp Y
                    a1 = (cX, cY)
                    cX += numbers[0]
                    cY += numbers[1] # d1
                    c1 = (cX, cY)
                    cX += numbers[2]
                    cY += numbers[3] # d2
                    c2 = (cX, cY)
                    cX += numbers[4]
                    cY += numbers[5]  # d3
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
                    a1 = (cX, cY)
                    cX += numbers[6]
                    cY += numbers[7] # d4
                    c1 = (cX, cY)
                    cX += numbers[8]
                    cY += numbers[9] # d5
                    c2 = (cX, cY)
                    if abs(cX - t_X) > abs(cY - t_Y):
                        cX = numbers[10]
                        cY = t_Y
                    else:
                        cX = t_X
                        cY = numbers[10]
                    a2 = (cX, cY)
                    control_points.extend(cX, cY)
                    for pt in cubic_bezier(a1, c1, c2, a2):
                        paths.extend(pt[0], pt[1])
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
                raise Exception("Cannot end table list without an operator (specifically, an endchar)")
            
        return paths.get_paths(), control_points.get_paths()

    def draw(self, display : bool = True, filename : str = None, plot_outline : bool = False,
                    plot_control_points : bool = False):
        paths, control_points = self.get_paths()

        curves = []
        for path in paths:
            if path[0] != path[-1]:
                path.append(path[0])
            curves.append(path)
            if plot_outline:
                plt.plot(*zip(*path))
        v = []
        c = []
        for crv in curves:
            pth = mp.Path(crv, closed=True)
            v += pth.vertices.tolist()
            c += pth.codes.tolist()
        compound_path = mp.Path(v, c)
        patch = PathPatch(compound_path, facecolor='black', edgecolor=None)
        plt.gca().add_patch(patch)
        plt.gca().plot()

        if plot_control_points: # plot control points?
            for path in control_points:
                plt.scatter(*zip(*path))
        plt.gca().set_aspect('equal')
        
        if filename:
            plt.savefig(filename)
        if display:
            plt.show()
        # TODO: matplotlib stuff


if __name__ == "__main__":
    table_list = [-1,426,338,'rmoveto',-1,-1,0,2,'rlineto',-3,-15,'rmoveto',0,4,0,3,-1,3,-1,7,-1,4,-637,-137,-7,-3,-6,-3,836,-3,11,-969,-920,-102,638,25,91,5,55,4,20,4,-587,-861,-83,-20,-5,565,'rrcurveto',59,4,31,-7,3,-19,2,-18,39,120,-13,61,-7,36,26,22,59,9,'rrcurveto',-388,-824,'rmoveto',8,-15,17,-9,25,-3,-1228,64,-4,67,38,55,10,14,-261,196,618,1121,-3,28,-1071,168,89,202,13,14,51,128,97,253,19,48,16,29,-3,15,-24,75,-8,28,9,15,'rrcurveto',-2,3,-1,2,-1,1,-35,16,-18,15,2,13,4,20,65,113,121,295,89,200,63,35,37,25,12,8,13,5,15,3,3,1,4,0,5,0,'rrcurveto',5,0,4,0,3,1,'rrcurveto',3,'hlineto',1,1,1,0,2,0,2,0,2,0,1,0,10,-2,17,-5,23,-8,70,-23,-38,-136,-6,-238,7,-67,10,67,14,0,1204,4,13,-14,10,-17,'rrcurveto',15,-27,0,-19,-15,-960,-67,40,-1159,1192,-243,103,-46,19,-51,331,1,-20,-16,-5,-23,-7,-30,-8,-5,-1,-5,-1,-4,-2,'rrcurveto',-12,-6,'rlineto',-8,-4,-5,-2,0,0,0,-6,3,1464,930,1147,5,-1227,-9,-369,1,-25,5,71,10,492,13,-403,'rrcurveto',192,-73,'rmoveto',-5,106,-10,94,-12,65,-1442,1251,74,-840,149,7,10,428,57,453,104,273,7,19,-4,15,-16,10,-7,3,3,9,-15,3,-13,3,4,5,17,3,'rrcurveto',7,3,5,-1483,-2,1245,'rrcurveto',5,842,'rmoveto',-5,5,'rlineto',-1,2,-2,1,-2,0,-4,0,-3,-3,-1,-3,'rrcurveto',6,-457,'rmoveto',5,3,-3,-7,'hvcurveto',0,5,-3,3,-7,0,-8,0,-2,-5,4,-7,'rrcurveto',5,70,'rmoveto',-5,-3,402,68,'hvcurveto',9,5,8,10,-4,14,'rrcurveto',73,-300,137,-27,16,-11,23,18,-3,6,-1,4,0,2,0,7,-81,-11,-56,-32,329,0,5,-435,52,-312,95,'blend',-103,54,-122,72,-140,119,-54,47,-30,-28,190,317,'rrcurveto',-334,-548,'rmoveto',98,-51,89,-220,79,-183,42,-94,24,-26,7,72,'rrcurveto',1006,49,'rmoveto',70,65,'rlineto',16,14,-117,-234,-5,-957,'rrcurveto',-15,-12,'rlineto',70,-89,46,5,-2,-39,-4,225,12,160,-7,107,'rrcurveto',-1470,-568,'rmoveto',-24,-10,-19,1,-13,12,'rrcurveto',19,-840,14,5,'rlineto',39,-6,27,8,15,20,'rrcurveto',-584,398,'rmoveto',14,-1,'rlineto',619,-108,337,647,16,54,-43,9,-135,-54,29,210,'rrcurveto',112,192,'rmoveto',-11,827,104,18,5,-28,-59,-22,-12,'endchar']
    # table_list = [-71,660,-21,'rmoveto',-464,'hlineto',13,27,7,29,0,31,0,33,-7,30,-13,27,'rrcurveto',-166,328,155,291,'rlineto',20,37,10,42,0,46,0,33,-6,29,-13,26,'rrcurveto',779,'hlineto',-26,-27,-18,-39,-11,-52,-5,-26,-3,-24,0,-21,'rrcurveto',-630,'vlineto',0,-85,21,-63,42,-41,'rrcurveto',-189,255,'hlineto',-97,-192,-1,0,'rlineto',-94,62,'rmoveto',192,377,0,380,-413,0,-216,-382,190,-374,'rlineto','endchar']
    # table_list = [1024,82,'rmoveto',614,'vlineto',0,59,-22,51,-44,44,'rrcurveto',-61,61,'rlineto',-44,75,-65,38,-86,0,'rrcurveto',-552,'hlineto',-45,0,-39,-16,-32,-32,-32,-32,-16,-39,0,-45,0,-64,29,-48,57,-33,'rrcurveto',69,-68,'rlineto',-1,1,-1,1,'vhcurveto',-47,-15,-38,-27,-29,-40,-29,-40,-14,-44,0,-49,'rrcurveto',-246,'vlineto',0,-89,37,-65,75,-41,'rrcurveto',62,-62,'rlineto',43,-43,51,-22,59,0,'rrcurveto',492,'hlineto',59,0,51,22,43,43,43,43,22,52,0,60,'rrcurveto',-137,102,'rmoveto',0,-51,-18,-43,-36,-36,-36,-36,-44,-18,-51,0,'rrcurveto',-491,'hlineto',-51,0,-43,18,-36,36,-36,36,-18,43,0,51,'rrcurveto',246,'vlineto',0,51,18,44,36,36,36,36,44,18,51,0,'rrcurveto',430,123,-491,'hlineto',-35,0,-29,12,-24,24,-24,24,-12,29,0,35,0,35,12,29,24,24,24,24,29,12,35,0,'rrcurveto',552,'hlineto',51,0,44,-18,36,-36,36,-36,18,-43,0,-51,'rrcurveto',-246,-553,'rmoveto',184,-369,-184,'vlineto',328,41,'rmoveto',-190,6,190,'hlineto','endchar']
    # table_list = [137,827,694,'rmoveto',23,27,12,31,0,35,0,51,-24,24,-47,-4,'rrcurveto',-197,-16,'rlineto',-29,-3,-43,-1,-56,1,-60,1,-42,3,-25,4,-19,3,-17,-12,-15,-27,-11,-19,-7,-22,-4,-24,-3,-15,-1,-13,0,-12,0,-42,12,-20,25,3,'rrcurveto',5,1,6,1,7,1,-15,-47,-20,-45,-26,-43,-26,-43,-41,-56,-55,-69,-61,-23,-34,-31,-7,-39,-9,-51,-4,-71,0,-91,0,-58,4,-29,7,1,'rrcurveto',4,1,10,0,16,0,11,-1,10,1,9,3,27,1,26,25,26,49,'rrcurveto',50,110,'rlineto',3,9,2,6,1,3,163,5,105,2,48,-2,23,-1,11,-24,0,-47,0,-5,0,-7,-1,-9,'rrcurveto',-21,'vlineto',0,-33,4,-24,7,-15,10,-21,19,-9,27,2,29,1,24,12,19,22,7,7,14,46,21,85,19,72,12,47,4,23,1,4,0,4,0,4,'rrcurveto',0,38,-23,25,-45,13,11,83,9,61,7,39,9,51,9,40,9,29,13,1,10,4,6,8,6,8,6,8,6,8,'rrcurveto',-221,-29,'rmoveto',-13,-66,-10,-50,-7,-33,-17,-71,-14,-56,-11,-41,-51,-3,-67,-1,-83,0,37,55,29,50,21,46,19,41,20,55,21,69,59,2,42,2,25,1,'rrcurveto','endchar']
    # table_list = [11,548,864,'rmoveto',-288,-48,72,-96,0,-48,-96,48,'rlineto',-80,-11,-40,-24,0,-37,'rrcurveto',-48,288,'vlineto',96,72,48,0,120,-24,0,-48,-48,-72,'rlineto',-400,-200,-112,-224,'hvcurveto',27,-128,64,-64,101,0,'rrcurveto',48,'hlineto',136,0,104,16,72,32,'rrcurveto',-48,120,-72,0,'rlineto',-64,-16,-32,-32,'vhcurveto',-24,0,-48,48,0,24,'rlineto',0,124,104,80,208,36,'rrcurveto',0,-24,-48,-144,24,0,24,-24,-24,-168,120,-24,216,0,-24,72,48,576,'rlineto',144,-128,72,-256,'vhcurveto',-456,-648,'rmoveto',72,0,48,-48,0,-48,'rlineto',-80,11,-40,24,0,37,'rrcurveto',576,24,'rmoveto',96,0,-24,-144,-24,0,-48,48,'rlineto','endchar']
    # table_list = [53,38,'hmoveto',204,0,-18,163,'rlineto',0,1,0,1,-1,1,-1,0,-2,0,-2,0,-2,0,-1,0,-1,0,'rrcurveto',-1,4,-32,325,-24,31,'rlineto',-56,0,-28,0,-1,-1,-1,-1,0,-1,0,-1,-3,-3,-6,-8,-9,-13,'rrcurveto',-2,-3,'rlineto',0,-1,0,-4,-1,-8,-1,-8,-1,-7,-1,-5,'rrcurveto',-11,46,'vlineto',3,11,43,0,'rlineto',0,-3,0,-4,0,-6,0,-6,0,-7,0,-7,0,-7,0,-7,0,-6,0,-6,0,-4,0,-3,'rrcurveto',-7,-76,'vlineto',-29,-45,-25,-253,'rlineto',78,199,'rmoveto',55,0,1,-11,'rlineto',0,-1,0,-4,0,-6,0,-6,0,-7,0,-8,0,-8,0,-8,0,-9,0,-9,0,-7,0,-5,1,-2,0,-6,0,-9,'rrcurveto',-16,-63,18,'vlineto',0,4,0,7,0,10,0,10,0,10,1,11,1,11,0,10,0,10,0,10,0,7,0,4,'rrcurveto','endchar']
    # table_list = [2,303,67,'rmoveto',-4,-32,-3,-31,'rlineto',1,-6,9,-3,16,0,'rrcurveto',31,1,51,-8,'rlineto',19,7,16,4,13,0,'rrcurveto',12,-1,14,-2,5,3,5,17,'rlineto',1,6,-7,4,-16,3,-16,3,-10,3,-5,3,-5,6,-4,11,-3,16,-9,51,-6,39,-3,27,11,26,5,46,0,67,0,38,-5,26,-9,15,'rrcurveto',-6,10,-21,20,-35,29,'rrcurveto',-62,5,-95,-23,'rlineto',-77,-20,-38,-17,0,-15,0,-19,2,-14,4,-9,'rrcurveto',-10,-12,20,-12,-6,-10,-3,1,'rlineto',4,-1,5,0,7,0,14,0,8,4,1,8,7,27,11,18,16,10,16,10,22,5,29,0,'rrcurveto',51,-34,19,-42,-21,-49,-66,-8,'rlineto',-17,-5,-23,-11,-29,-17,'rrcurveto',-72,-48,'rlineto',-5,-4,-5,-12,-5,-21,-5,-21,-2,-15,0,-9,'rrcurveto',1,-58,'rlineto',15,-15,17,-14,20,-13,24,-15,20,-9,17,-2,'rrcurveto',58,11,50,49,'rlineto',7,6,7,6,8,6,8,6,6,4,4,3,'rrcurveto',-4,93,'rmoveto',-26,-60,-63,-35,-35,4,-18,55,'rlineto',9,27,11,25,13,24,'rrcurveto',46,4,'rlineto',17,13,21,14,25,15,'rrcurveto',-4,-8,1,-22,3,-19,'rlineto','endchar']
    viz = Visualizer(table_list)
    viz.draw(filename='ttt.png')