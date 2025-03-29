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

            if len(numbers) == 0 and operator != "endchar":
                continue

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
                print(running_idx)
                print(numbers)
                print(operator)
                raise Exception("Cannot end table list without an operator (specifically, an endchar)")
            
        return paths.get_paths(), control_points.get_paths()

    def draw(self, display : bool = True, filename : str = None, plot_outline : bool = False,
                    plot_control_points : bool = False, return_image : bool = False,
                    bounds : tuple = None, im_size_inches : tuple = None, center : bool = False,
                    dpi : int = 100):
        paths, control_points = self.get_paths()
        
        if center:
            min_x = min([min([vi[0] for vi in path]) for path in control_points])
            max_x = max([max([vi[0] for vi in path]) for path in control_points])
            min_y = min([min([vi[1] for vi in path]) for path in control_points])
            max_y = max([max([vi[1] for vi in path]) for path in control_points])
            mean_x = min_x + (max_x - min_x) // 2
            mean_y = min_y + (max_y - min_y) // 2

            for path_idx in range(len(paths)):
                for coord_idx in range(len(paths[path_idx])):
                    paths[path_idx][coord_idx] = (paths[path_idx][coord_idx][0] - mean_x, paths[path_idx][coord_idx][1] - mean_y)
            
            for path_idx in range(len(control_points)):
                for coord_idx in range(len(control_points[path_idx])):
                    control_points[path_idx][coord_idx] = (control_points[path_idx][coord_idx][0] - mean_x, control_points[path_idx][coord_idx][1] - mean_y)

        fig = plt.figure()
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
            if len(crv) < 2:
                continue
            pth = mp.Path(crv, closed=True)
            v += pth.vertices.tolist()
            c += pth.codes.tolist()
        
        compound_path = mp.Path(v, c)
        patch = PathPatch(compound_path, facecolor='black', edgecolor=None)
        fig.gca().add_patch(patch)
        fig.gca().plot()

        if plot_control_points: # plot control points?
            for path in control_points:
                plt.scatter(*zip(*path))
        fig.gca().set_aspect('equal')
        
        if filename:
            fig.savefig(filename)
        if display:
            # fig.show()
            plt.show()
        if return_image:
            fig.set_dpi(dpi)
            fig.gca().axis('off')
            if bounds is not None:
                fig.gca().set_xlim([bounds[0], bounds[1]])
                fig.gca().set_ylim([bounds[0], bounds[1]])
            if im_size_inches is not None:
                fig.gca().figure.set_figwidth(im_size_inches[1])
                fig.gca().figure.set_figheight(im_size_inches[0])
            canvas = fig.gca().figure.canvas
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(*reversed(canvas.get_width_height()), 3)
            plt.close()
            return img
        else:
            plt.close()


if __name__ == "__main__":
    invert = False
    decumulate = False
    basic = True

    if invert or decumulate or basic:
        from tablelist_utils import numbers_first, operator_first, make_non_cumulative, use_basic_operators, sort_tablelist
        min_number = -1500
        max_number = 1500
        pad_token = "<PAD>"
        sos_token = "<SOS>"
        eos_token = "<EOS>"
        from tokenizer import Tokenizer
        from config import operators
        tokenizer = Tokenizer(
            min_number=min_number,
            max_number=max_number,
            possible_operators=operators,
            pad_token=pad_token,
            sos_token=sos_token,
            eos_token=eos_token
        )

    # table_list = [936,299,1429,'rmoveto',-45,-9,0,-18,'rlineto',40,0,71,-59,102,-118,-21,6,-12,3,-3,0,-103,112,-95,56,-86,0,-65,-8,-33,-29,0,-51,'rrcurveto',-36,'vlineto',65,-171,86,-86,106,0,91,17,53,9,15,0,'rrcurveto',18,'hlineto',23,0,18,-56,13,-112,0,-49,39,-89,77,-129,9,-295,29,-148,50,0,0,23,6,12,12,0,23,0,24,-12,25,-23,'rrcurveto',18,'hlineto',17,9,9,9,0,8,'rrcurveto',27,'vlineto',0,20,-9,21,-17,21,-20,0,-21,-9,-21,-18,0,21,-3,27,-6,32,41,53,53,27,66,0,'rrcurveto',17,-9,18,9,'rlineto',84,0,59,-30,35,-59,-6,-10,-3,-12,0,-13,'rrcurveto',-36,-9,'vlineto',-36,27,-17,0,'rlineto',-13,0,-12,-24,-11,-47,4,-29,9,-15,14,0,'rrcurveto',17,0,54,35,17,0,'rlineto',12,6,-15,-29,'hvcurveto',59,64,30,89,0,113,'rrcurveto',115,'vlineto',0,42,6,33,12,23,83,131,41,130,0,129,0,11,15,6,29,0,28,0,50,-9,73,-17,'rrcurveto',54,9,'rlineto',142,73,71,86,0,99,'rrcurveto',35,'vlineto',53,-33,27,-65,'vhcurveto',-65,0,-53,-12,-42,-23,-114,-101,-62,-50,-10,0,'rrcurveto',-9,9,'hlineto',15,0,38,38,62,77,15,15,42,27,69,39,'rrcurveto',9,-54,'vlineto',-154,-58,-77,-53,0,-49,'rrcurveto',-35,'hlineto',-16,7,-21,27,-25,46,-30,53,-83,27,-136,0,'rrcurveto',-97,'hlineto',-61,0,-47,-3,-33,-6,-62,0,-65,-50,-69,-101,'rrcurveto',-44,'hlineto',0,53,-77,53,-154,53,'rrcurveto',267,-479,'rmoveto',9,17,9,9,9,0,'rrcurveto',27,'hlineto',19,0,18,-29,17,-59,'rrcurveto',-27,-9,'vlineto',-9,65,-15,33,-21,0,'rrcurveto',-18,0,-35,-27,'rlineto',647,-89,'rmoveto',6,89,18,44,30,0,'rrcurveto',9,'hlineto',29,0,18,-9,6,-17,'rrcurveto',-18,'vlineto',-22,18,-21,9,-19,0,-30,-27,-15,-36,0,-45,'rrcurveto',36,-35,-45,-9,'rlineto',-630,106,'rmoveto',0,10,6,6,12,2,'rrcurveto',18,-9,'hlineto',-12,-6,-6,-12,'vhcurveto',657,27,'rmoveto',26,-27,-8,'hlineto',-10,0,-6,6,-2,12,'rrcurveto',-630,-98,'rmoveto',13,0,18,9,23,18,0,-13,-3,-12,-6,-10,'rrcurveto',-44,'hlineto',514,-684,'rmoveto',0,-15,-68,-12,-136,-8,-90,6,-62,9,-34,11,'rrcurveto',-9,'hlineto',17,-83,47,-41,78,0,'rrcurveto',62,9,'rlineto',31,-6,18,-3,4,0,101,16,50,44,0,73,'rrcurveto','endchar']
    # table_list = [107,-70,74,'rmoveto',2,9,3,10,4,10,4,11,4,10,5,10,5,11,4,9,4,10,4,10,4,9,3,9,'rrcurveto',59,'hlineto',3,-10,2,-13,4,-16,5,-16,4,-17,5,-19,4,-19,5,-19,6,-19,5,-20,6,-18,5,-16,6,-17,5,-15,5,-13,4,-14,5,-9,4,-6,'rrcurveto',16,-22,'rlineto',-2,-1,-5,0,-7,-1,-6,-1,-8,0,-8,0,-9,0,-7,0,-8,0,-7,0,-5,0,-2,0,'rrcurveto',-5,'hlineto',-4,0,-3,3,-3,6,-3,5,-2,4,0,3,-2,6,-3,6,-1,4,-1,4,-2,6,-3,7,'rrcurveto',-6,7,-67,0,'rlineto',-6,0,-4,-3,-2,-7,-2,-6,-3,-7,-3,-9,-4,-8,-4,-8,-7,-7,-6,-8,-10,-5,-15,-2,'rrcurveto',-5,'hlineto',-6,0,-8,1,-11,1,-10,2,-8,1,-6,2,'rrcurveto',158,196,'rmoveto',-1,-3,-1,-3,-3,-5,-3,-5,-2,-5,-4,-4,-3,-5,-3,-5,-3,-5,-3,-5,-1,-3,-1,-2,0,-1,-2,-2,-1,-3,-2,-4,0,-2,-1,-1,'rrcurveto',1,0,2,0,3,0,4,0,2,0,2,0,'rrcurveto',10,'hlineto',6,0,6,2,5,2,4,2,2,4,0,7,'rrcurveto',11,'vlineto',0,8,-1,6,-2,7,-2,7,-3,5,-5,2,'rrcurveto','endchar']
    # table_list = [107,133,274,'rmoveto',3,13,4,14,6,15,6,15,6,15,7,15,7,15,6,14,6,14,6,14,5,14,4,13,'rrcurveto',88,'hlineto',3,-15,4,-19,6,-23,6,-23,6,-25,7,-27,7,-27,7,-28,8,-28,8,-28,8,-26,8,-25,8,-25,7,-22,7,-19,7,-19,6,-14,6,-8,'rrcurveto',23,-32,'rlineto',-3,-1,-6,-1,-10,-1,-10,-1,-11,0,-12,0,-12,0,-11,0,-11,0,-11,0,-7,0,-3,0,'rrcurveto',-6,'hlineto',-6,0,-5,4,-4,8,-4,8,-3,6,-1,4,-3,9,-3,8,-2,6,-2,6,-3,8,-4,10,'rrcurveto',-9,11,-98,0,'rlineto',-8,0,-6,-5,-3,-9,-3,-9,-4,-11,-5,-12,-5,-12,-7,-11,-9,-11,-9,-11,-15,-7,-21,-3,'rrcurveto',-7,'hlineto',-9,0,-12,1,-15,2,-15,2,-12,2,-9,3,'rrcurveto',229,284,'rmoveto',-1,-3,-2,-5,-4,-7,-4,-7,-4,-7,-5,-7,-5,-7,-4,-7,-4,-7,-4,-7,-2,-5,-1,-3,-1,-1,-2,-3,-2,-5,-2,-5,-1,-3,-1,-1,'rrcurveto',1,0,3,0,5,0,5,0,4,0,2,0,'rrcurveto',15,'hlineto',9,0,8,2,7,3,7,3,3,6,0,10,'rrcurveto',16,'vlineto',0,11,-1,10,-3,10,-3,10,-5,7,-8,3,'rrcurveto','endchar']
    # table_list = [107,-23,24,'rmoveto',1,3,1,4,1,3,1,4,2,3,1,3,2,4,1,3,2,3,1,4,1,3,1,3,'rrcurveto',19,'hlineto',1,-4,1,-4,1,-5,2,-6,1,-5,2,-7,1,-6,2,-6,2,-7,2,-6,2,-6,1,-5,2,-6,2,-5,2,-4,1,-5,2,-3,1,-2,'rrcurveto',5,-7,'rlineto',0,0,-2,0,-2,-1,-2,0,-3,0,-3,0,-3,0,-2,0,-3,0,-2,0,-2,0,0,0,'rrcurveto',-2,'hlineto',-1,0,-1,1,-1,2,-1,2,-1,1,0,1,-1,2,-1,2,0,1,0,2,-1,2,-1,2,'rrcurveto',-2,2,-22,0,'rlineto',-2,0,-1,-1,-1,-2,0,-2,-1,-2,-1,-3,-2,-3,-1,-3,-2,-2,-2,-3,-4,-1,-5,-1,'rrcurveto',-1,'hlineto',-2,0,-3,0,-4,1,-3,0,-3,1,-2,0,'rrcurveto',53,65,'rmoveto',0,-1,-1,-1,-1,-2,-1,-1,0,-2,-2,-1,-1,-2,-1,-2,-1,-1,-1,-2,0,-1,0,-1,0,0,-1,-1,0,-1,-1,-1,0,-1,0,0,'rrcurveto',0,0,1,0,1,0,1,0,1,0,0,0,'rrcurveto',4,'hlineto',2,0,2,1,1,0,1,1,1,1,0,3,'rrcurveto',3,'vlineto',0,3,-1,2,0,2,-1,3,0,1,-2,1,'rrcurveto','endchar']
    # table_list = ['hmoveto', 55, 3, 'hmoveto', 553, 'rlineto', 756, 0, 459, 697, 301, 697, 3, 0, 129, 0, 216, 205, 473, 205, 'rmoveto', 349, 521, 'rlineto', 423, 333, 269, 333, 'endchar']
    # table_list = ['rmoveto', 498, 394, 'hlineto', 499, 'rrcurveto', 498, 389, 499, 384, 501, 379, 505, 372, 508, 366, 508, 363, 'hlineto', 501, 'rrcurveto', 497, 367, 493, 369, 490, 370, 'hvcurveto', 488, 486, 371, 371, 'vlineto', 372, 'vhcurveto', 372, 493, 373, 499, 'rrcurveto', 499, 377, 499, 379, 501, 379, 505, 380, 506, 380, 504, 379, 'vlineto', 380, 'rrcurveto', 512, 375, 516, 370, 516, 365, 513, 377, 519, 381, 524, 380, 532, 384, 537, 388, 538, 390, 540, 395, 540, 401, 539, 404, 533, 418, 524, 425, 511, 429, 507, 433, 504, 436, 500, 440, 'rrcurveto', 501, 438, 503, 437, 505, 437, 510, 438, 513, 437, 514, 437, 513, 436, 509, 437, 502, 441, 498, 446, 494, 450, 493, 451, 489, 453, 403, 454, 365, 452, 332, 452, 305, 454, 285, 455, 'rrcurveto', 266, 456, 246, 447, 222, 433, 206, 420, 199, 409, 198, 407, 196, 404, 195, 400, 196, 392, 'rlineto', 173, 379, 'rrcurveto', 165, 375, 162, 370, 166, 364, 'rlineto', 140, 357, 'rrcurveto', 148, 356, 152, 356, 151, 359, 'rlineto', 139, 363, 'rrcurveto', 131, 364, 125, 367, 121, 372, 'rlineto', 103, 390, 'rrcurveto', 98, 392, 94, 392, 90, 391, 85, 389, 82, 389, 80, 388, 'rlineto', 67, 390, 53, 386, 'rrcurveto', 47, 383, 42, 379, 38, 376, 'rlineto', 33, 370, 'rrcurveto', 29, 366, 23, 365, 15, 363, 'rlineto', -3, 361, 'rrcurveto', -12, 355, -22, 349, -32, 343, -37, 335, -40, 327, -40, 319, 'vlineto', 315, 'rrcurveto', -38, 316, -35, 318, -32, 322, -28, 327, -24, 330, -22, 329, 'rlineto', -10, 318, 'rrcurveto', -18, 319, -18, 321, -16, 324, -12, 330, -12, 335, -15, 338, 'rlineto', -17, 332, 'rrcurveto', -25, 326, -31, 322, -37, 321, 'rlineto', -31, 317, -27, 313, 'rrcurveto', -22, 312, -20, 307, -20, 299, 'vlineto', 297, 'rrcurveto', -21, 296, -23, 295, -26, 295, -27, 297, -29, 297, -31, 296, -35, 295, -44, 289, -59, 278, 'rlineto', -53, 254, 'rrcurveto', -54, 246, -50, 238, -48, 229, 'rlineto', -12, 210, 32, 182, 'vhcurveto', 180, 29, 178, 29, 'rlineto', 33, 175, 'rrcurveto', 36, 175, 39, 174, 47, 175, 'rlineto', 74, 174, 75, 177, 'rrcurveto', 71, 184, 66, 193, 60, 201, 'rlineto', 56, 204, 'rrcurveto', 53, 205, 52, 204, 53, 204, 55, 206, 59, 205, 67, 204, 'rlineto', 67, 205, 73, 214, 'rrcurveto', 87, 207, 98, 202, 108, 198, 111, 195, 121, 193, 135, 194, 123, 192, 116, 193, 111, 196, 'rlineto', 94, 208, 'rrcurveto', 94, 207, 95, 206, 97, 205, 'rlineto', 106, 223, 'rrcurveto', 104, 226, 102, 232, 97, 243, 'rlineto', 101, 258, 'rrcurveto', 104, 262, 106, 265, 107, 267, 'hlineto', 158, 'rrcurveto', 156, 259, 156, 246, 157, 223, 'rlineto', 173, 176, 'rrcurveto', 181, 181, 187, 190, 190, 201, 229, 202, 267, 200, 308, 196, 350, 193, 385, 187, 412, 184, 'hlineto', 415, 'rrcurveto', 421, 185, 428, 185, 431, 185, 435, 185, 436, 184, 433, 181, 431, 176, 433, 173, 437, 171, 443, 173, 448, 172, 452, 169, 457, 163, 462, 156, 462, 148, 458, 152, 454, 155, 447, 156, 'rrcurveto', 437, 158, 430, 157, 424, 156, 410, 155, 399, 151, 394, 145, 390, 142, 389, 140, 391, 139, 398, 130, 409, 122, 423, 118, 'hlineto', 429, 'rrcurveto', 436, 118, 446, 117, 457, 116, 466, 115, 469, 113, 471, 109, 472, 106, 471, 103, 469, 98, 'rlineto', 464, 89, 466, 84, 467, 79, 'rrcurveto', 474, 78, 480, 78, 484, 80, 492, 85, 494, 90, 491, 95, 474, 99, 456, 100, 440, 98, 425, 97, 417, 93, 420, 87, 'rlineto', 427, 82, 'rrcurveto', 440, 81, 453, 78, 462, 74, 'rlineto', 465, 85, 460, 87, 'rrcurveto', 458, 90, 455, 89, 451, 83, 'vlineto', 83, 'rrcurveto', 450, 82, 451, 78, 452, 73, 455, 64, 455, 56, 451, 51, 447, 46, 442, 40, 438, 32, 434, 25, 427, 19, 418, 14, 'rlineto', 411, 8, 'rrcurveto', 410, 8, 409, 7, 409, 6, 408, 6, 408, 8, 410, 9, 415, 21, 418, 32, 421, 46, 426, 63, 436, 74, 450, 79, 'rlineto', 477, 84, 'rrcurveto', 478, 90, 479, 93, 480, 93, 484, 92, 490, 90, 496, 84, 497, 88, 498, 90, 500, 88, 'rlineto', 488, 93, 'rrcurveto', 488, 100, 484, 100, 473, 95, 'rlineto', 492, 84, 499, 84, 536, 79, 511, 83, 'rrcurveto', 516, 72, 516, 63, 515, 55, 'rlineto', 542, 32, 550, 31, 585, 51, 'rmoveto', 107, 222, 'rlineto', 113, 224, 'rrcurveto', 117, 229, 118, 235, 119, 241, 121, 255, 125, 267, 132, 277, 'rlineto', 174, 284, 183, 295, 'rrcurveto', 189, 301, 196, 307, 201, 313, 206, 317, 213, 323, 220, 329, 218, 332, 216, 336, 213, 339, 'rlineto', 214, 348, 'rrcurveto', 218, 355, 222, 366, 228, 381, 234, 399, 242, 417, 254, 434, 259, 433, 267, 432, 277, 434, 'vlineto', 432, 'rrcurveto', 273, 415, 269, 400, 263, 388, 257, 376, 250, 363, 242, 350, 234, 335, 227, 743, 222, 748, 'vlineto', 745, 'rrcurveto', 221, 748, 218, 746, 214, 742, 211, 735, 200, 735, 180, 738, 166, 740, 155, 745, 148, 757, 141, 764, 139, 772, 146, 776, 107, 768, 83, 755, 72, 738, 63, 724, 60, 710, 62, 688, 67, 676, 81, 683, 100, 691, 'rrcurveto', 98, 693, 100, 694, 102, 694, 105, 690, 106, 684, 105, 677, 98, 679, 93, 685, 85, 693, 80, 700, 76, 703, 73, 704, 69, 709, 63, 715, 57, 722, 49, 726, 50, 736, 51, 749, 'rlineto', 54, 754, 'rrcurveto', 57, 752, 67, 749, 84, 746, 'endchar']
    # table_list = ['rmoveto', 52, 869, 432, 'rrcurveto', 874, 485, 867, 520, 843, 551, 819, 587, 791, 616, 762, 646, 729, 673, 692, 692, 645, 706, 598, 717, 554, 726, 510, 732, 462, 739, 418, 743, 365, 740, 314, 734, 263, 723, 211, 706, 'rrcurveto', 137, 693, 91, 679, 53, 663, 20, 646, 0, 632, -10, 616, -18, 598, -22, 581, -22, 564, -23, 551, -23, 539, -23, 530, -21, 509, -17, 492, -11, 474, -6, 459, -1, 440, 6, 417, 'rrcurveto', 15, 390, 28, 362, 41, 334, 51, 311, 62, 290, 73, 269, 85, 247, 98, 225, 113, 202, 131, 175, 151, 151, 172, 128, 184, 116, 197, 103, 209, 91, 216, 93, 221, 96, 225, 99, 'rrcurveto', 232, 101, 239, 104, 244, 108, 246, 106, 250, 104, 254, 100, 256, 99, 259, 98, 262, 96, 267, 97, 271, 99, 273, 98, 'rlineto', 283, 104, 'rrcurveto', 289, 107, 295, 109, 299, 109, 305, 109, 313, 111, 326, 114, 334, 115, 343, 116, 355, 119, 368, 119, 382, 120, 394, 120, 412, 121, 430, 121, 448, 120, 467, 119, 485, 119, 500, 120, 'rrcurveto', 515, 120, 524, 120, 530, 119, 538, 119, 546, 119, 553, 119, 563, 120, 573, 120, 585, 119, 603, 120, 617, 121, 630, 121, 640, 124, 654, 128, 669, 130, 680, 129, 687, 128, 691, 124, 'rrcurveto', 696, 121, 702, 116, 710, 109, 718, 101, 727, 91, 737, 81, 751, 74, 764, 64, 775, 51, 787, 38, 798, 26, 807, 13, 813, 7, 820, 5, 828, 6, 836, 6, 841, 8, 844, 13, 'rrcurveto', 852, 28, 856, 44, 856, 61, 854, 77, 852, 95, 849, 115, 846, 137, 837, 160, 831, 182, 827, 186, 827, 188, 826, 186, 827, 179, 831, 172, 838, 166, 843, 160, 851, 150, 857, 137, 'rrcurveto', 867, 123, 876, 108, 887, 93, 904, 76, 913, 64, 917, 56, 924, 51, 924, 48, 923, 47, 924, 45, 921, 42, 915, 38, 907, 44, 897, 50, 888, 57, 875, 62, 867, 71, 866, 82, 'rrcurveto', 867, 98, 872, 114, 878, 126, 880, 126, 876, 125, 868, 123, 861, 121, 853, 119, 842, 116, 831, 113, 822, 112, 815, 108, 805, 108, 795, 109, 782, 110, 769, 111, 757, 113, 740, 118, 733, 124, 726, 134, 724, 146, 'rrcurveto', 725, 162, 727, 179, 734, 198, 741, 218, 750, 237, 758, 254, 767, 270, 776, 282, 781, 292, 788, 299, 794, 305, 796, 308, 797, 309, 799, 308, 801, 308, 804, 308, 809, 307, 816, 305, 'rrcurveto', 828, 301, 834, 297, 837, 294, 840, 291, 842, 288, 842, 287, 'rlineto', 842, 282, 'rrcurveto', 716, 275, 747, 262, 764, 242, 784, 222, 795, 201, 797, 178, 803, 153, 801, 135, 796, 120, 788, 107, 780, 99, 773, 94, 764, 89, 760, 87, 759, 85, 766, 63, 771, 45, 776, 27, 'rrcurveto', 790, 5, 807, -10, 827, -22, 839, -22, 849, -16, 856, -5, 864, 4, 871, 14, 870, 25, 871, 33, 868, 40, 862, 49, 854, 60, 845, 69, 833, 75, 'rlineto', 793, 108, 'rrcurveto', 791, 113, 782, 119, 765, 128, 753, 138, 745, 147, 736, 154, 727, 162, 718, 169, 709, 173, 'rlineto', 710, 177, 670, -137, 'rrcurveto', 674, -139, 680, -142, 687, -141, 692, -141, 698, -141, 704, -147, 716, -154, 728, -158, 739, -161, 749, -163, 755, -170, 761, -183, 765, -77, 769, -7, 770, 33, 768, 30, 762, 25, 749, 15, 741, 6, 738, 3, 737, 4, 'rrcurveto', 729, 4, 723, 7, 716, 16, 712, 23, 706, 32, 699, 43, 693, 57, 689, 72, 685, 85, 678, 99, 671, 112, 667, 124, 663, 135, 658, 147, 654, 163, 651, 169, 650, 175, 649, 182, 'rrcurveto', 651, 188, 653, 194, 656, 199, 660, 203, 664, 210, 665, 218, 671, 232, 673, 245, 671, 256, 664, 248, 655, 241, 642, 234, 630, 226, 617, 219, 605, 211, 595, 203, 586, 195, 578, 187, 'rrcurveto', 573, 182, 568, 177, 566, 169, 563, 164, 560, 157, 557, 148, 550, 128, 540, 110, 529, 93, 519, 78, 507, 66, 496, 54, 484, 43, 477, 34, 471, 27, 464, 20, 458, 13, 453, 7, 'rrcurveto', 448, -1, 443, -7, 435, -11, 429, -16, 420, -19, 411, -19, 401, -20, 390, -16, 379, -7, 370, 3, 364, 16, 362, 31, 363, 45, 361, 57, 358, 68, 354, 80, 352, 90, 349, 100, 'rrcurveto', 347, 108, 341, 120, 331, 134, 320, 148, 307, 162, 292, 175, 278, 187, 263, 197, 247, 205, 235, 214, 223, 221, 210, 226, 198, 231, 184, 236, 172, 244, 159, 248, 147, 255, 137, 265, 'rrcurveto', 128, 278, 120, 292, 109, 308, 100, 323, 93, 340, 84, 356, 74, 370, 66, 388, 59, 413, 53, 435, 49, 458, 46, 477, 43, 495, 41, 511, 42, 533, 46, 556, 56, 576, 71, 603, 'rrcurveto', 92, 632, 121, 661, 157, 691, 200, 725, 248, 746, 301, 754, 335, 760, 372, 760, 413, 749, 450, 738, 480, 722, 507, 701, 529, 685, 554, 669, 576, 656, 610, 641, 638, 626, 659, 610, 'rrcurveto', 599, 702, 588, 721, 576, 735, 567, 750, 553, 759, 533, 771, 511, 781, 488, 785, 466, 788, 443, 786, 421, 778, 397, 762, 378, 749, 365, 738, 352, 727, 340, 716, 326, 705, 315, 697, 298, 687, 288, 677, 285, 672, 'endchar']
    # table_list = [-1,426,338,'rmoveto',-1,-1,0,2,'rlineto',-3,-15,'rmoveto',0,4,0,3,-1,3,-1,7,-1,4,-637,-137,-7,-3,-6,-3,836,-3,11,-969,-920,-102,638,25,91,5,55,4,20,4,-587,-861,-83,-20,-5,565,'rrcurveto',59,4,31,-7,3,-19,2,-18,39,120,-13,61,-7,36,26,22,59,9,'rrcurveto',-388,-824,'rmoveto',8,-15,17,-9,25,-3,-1228,64,-4,67,38,55,10,14,-261,196,618,1121,-3,28,-1071,168,89,202,13,14,51,128,97,253,19,48,16,29,-3,15,-24,75,-8,28,9,15,'rrcurveto',-2,3,-1,2,-1,1,-35,16,-18,15,2,13,4,20,65,113,121,295,89,200,63,35,37,25,12,8,13,5,15,3,3,1,4,0,5,0,'rrcurveto',5,0,4,0,3,1,'rrcurveto',3,'hlineto',1,1,1,0,2,0,2,0,2,0,1,0,10,-2,17,-5,23,-8,70,-23,-38,-136,-6,-238,7,-67,10,67,14,0,1204,4,13,-14,10,-17,'rrcurveto',15,-27,0,-19,-15,-960,-67,40,-1159,1192,-243,103,-46,19,-51,331,1,-20,-16,-5,-23,-7,-30,-8,-5,-1,-5,-1,-4,-2,'rrcurveto',-12,-6,'rlineto',-8,-4,-5,-2,0,0,0,-6,3,1464,930,1147,5,-1227,-9,-369,1,-25,5,71,10,492,13,-403,'rrcurveto',192,-73,'rmoveto',-5,106,-10,94,-12,65,-1442,1251,74,-840,149,7,10,428,57,453,104,273,7,19,-4,15,-16,10,-7,3,3,9,-15,3,-13,3,4,5,17,3,'rrcurveto',7,3,5,-1483,-2,1245,'rrcurveto',5,842,'rmoveto',-5,5,'rlineto',-1,2,-2,1,-2,0,-4,0,-3,-3,-1,-3,'rrcurveto',6,-457,'rmoveto',5,3,-3,-7,'hvcurveto',0,5,-3,3,-7,0,-8,0,-2,-5,4,-7,'rrcurveto',5,70,'rmoveto',-5,-3,402,68,'hvcurveto',9,5,8,10,-4,14,'rrcurveto',73,-300,137,-27,16,-11,23,18,-3,6,-1,4,0,2,0,7,-81,-11,-56,-32,329,0,5,-435,52,-312,95,'blend',-103,54,-122,72,-140,119,-54,47,-30,-28,190,317,'rrcurveto',-334,-548,'rmoveto',98,-51,89,-220,79,-183,42,-94,24,-26,7,72,'rrcurveto',1006,49,'rmoveto',70,65,'rlineto',16,14,-117,-234,-5,-957,'rrcurveto',-15,-12,'rlineto',70,-89,46,5,-2,-39,-4,225,12,160,-7,107,'rrcurveto',-1470,-568,'rmoveto',-24,-10,-19,1,-13,12,'rrcurveto',19,-840,14,5,'rlineto',39,-6,27,8,15,20,'rrcurveto',-584,398,'rmoveto',14,-1,'rlineto',619,-108,337,647,16,54,-43,9,-135,-54,29,210,'rrcurveto',112,192,'rmoveto',-11,827,104,18,5,-28,-59,-22,-12,'endchar']
    table_list = [-71,660,-21,'rmoveto',-464,'hlineto',13,27,7,29,0,31,0,33,-7,30,-13,27,'rrcurveto',-166,328,155,291,'rlineto',20,37,10,42,0,46,0,33,-6,29,-13,26,'rrcurveto',779,'hlineto',-26,-27,-18,-39,-11,-52,-5,-26,-3,-24,0,-21,'rrcurveto',-630,'vlineto',0,-85,21,-63,42,-41,'rrcurveto',-189,255,'hlineto',-97,-192,-1,0,'rlineto',-94,62,'rmoveto',192,377,0,380,-413,0,-216,-382,190,-374,'rlineto','endchar']
    # table_list = [1024,82,'rmoveto',614,'vlineto',0,59,-22,51,-44,44,'rrcurveto',-61,61,'rlineto',-44,75,-65,38,-86,0,'rrcurveto',-552,'hlineto',-45,0,-39,-16,-32,-32,-32,-32,-16,-39,0,-45,0,-64,29,-48,57,-33,'rrcurveto',69,-68,'rlineto',-1,1,-1,1,'vhcurveto',-47,-15,-38,-27,-29,-40,-29,-40,-14,-44,0,-49,'rrcurveto',-246,'vlineto',0,-89,37,-65,75,-41,'rrcurveto',62,-62,'rlineto',43,-43,51,-22,59,0,'rrcurveto',492,'hlineto',59,0,51,22,43,43,43,43,22,52,0,60,'rrcurveto',-137,102,'rmoveto',0,-51,-18,-43,-36,-36,-36,-36,-44,-18,-51,0,'rrcurveto',-491,'hlineto',-51,0,-43,18,-36,36,-36,36,-18,43,0,51,'rrcurveto',246,'vlineto',0,51,18,44,36,36,36,36,44,18,51,0,'rrcurveto',430,123,-491,'hlineto',-35,0,-29,12,-24,24,-24,24,-12,29,0,35,0,35,12,29,24,24,24,24,29,12,35,0,'rrcurveto',552,'hlineto',51,0,44,-18,36,-36,36,-36,18,-43,0,-51,'rrcurveto',-246,-553,'rmoveto',184,-369,-184,'vlineto',328,41,'rmoveto',-190,6,190,'hlineto','endchar']
    # table_list = [137,827,694,'rmoveto',23,27,12,31,0,35,0,51,-24,24,-47,-4,'rrcurveto',-197,-16,'rlineto',-29,-3,-43,-1,-56,1,-60,1,-42,3,-25,4,-19,3,-17,-12,-15,-27,-11,-19,-7,-22,-4,-24,-3,-15,-1,-13,0,-12,0,-42,12,-20,25,3,'rrcurveto',5,1,6,1,7,1,-15,-47,-20,-45,-26,-43,-26,-43,-41,-56,-55,-69,-61,-23,-34,-31,-7,-39,-9,-51,-4,-71,0,-91,0,-58,4,-29,7,1,'rrcurveto',4,1,10,0,16,0,11,-1,10,1,9,3,27,1,26,25,26,49,'rrcurveto',50,110,'rlineto',3,9,2,6,1,3,163,5,105,2,48,-2,23,-1,11,-24,0,-47,0,-5,0,-7,-1,-9,'rrcurveto',-21,'vlineto',0,-33,4,-24,7,-15,10,-21,19,-9,27,2,29,1,24,12,19,22,7,7,14,46,21,85,19,72,12,47,4,23,1,4,0,4,0,4,'rrcurveto',0,38,-23,25,-45,13,11,83,9,61,7,39,9,51,9,40,9,29,13,1,10,4,6,8,6,8,6,8,6,8,'rrcurveto',-221,-29,'rmoveto',-13,-66,-10,-50,-7,-33,-17,-71,-14,-56,-11,-41,-51,-3,-67,-1,-83,0,37,55,29,50,21,46,19,41,20,55,21,69,59,2,42,2,25,1,'rrcurveto','endchar']
    # table_list = [11,548,864,'rmoveto',-288,-48,72,-96,0,-48,-96,48,'rlineto',-80,-11,-40,-24,0,-37,'rrcurveto',-48,288,'vlineto',96,72,48,0,120,-24,0,-48,-48,-72,'rlineto',-400,-200,-112,-224,'hvcurveto',27,-128,64,-64,101,0,'rrcurveto',48,'hlineto',136,0,104,16,72,32,'rrcurveto',-48,120,-72,0,'rlineto',-64,-16,-32,-32,'vhcurveto',-24,0,-48,48,0,24,'rlineto',0,124,104,80,208,36,'rrcurveto',0,-24,-48,-144,24,0,24,-24,-24,-168,120,-24,216,0,-24,72,48,576,'rlineto',144,-128,72,-256,'vhcurveto',-456,-648,'rmoveto',72,0,48,-48,0,-48,'rlineto',-80,11,-40,24,0,37,'rrcurveto',576,24,'rmoveto',96,0,-24,-144,-24,0,-48,48,'rlineto','endchar']
    # table_list = [53,38,'hmoveto',204,0,-18,163,'rlineto',0,1,0,1,-1,1,-1,0,-2,0,-2,0,-2,0,-1,0,-1,0,'rrcurveto',-1,4,-32,325,-24,31,'rlineto',-56,0,-28,0,-1,-1,-1,-1,0,-1,0,-1,-3,-3,-6,-8,-9,-13,'rrcurveto',-2,-3,'rlineto',0,-1,0,-4,-1,-8,-1,-8,-1,-7,-1,-5,'rrcurveto',-11,46,'vlineto',3,11,43,0,'rlineto',0,-3,0,-4,0,-6,0,-6,0,-7,0,-7,0,-7,0,-7,0,-6,0,-6,0,-4,0,-3,'rrcurveto',-7,-76,'vlineto',-29,-45,-25,-253,'rlineto',78,199,'rmoveto',55,0,1,-11,'rlineto',0,-1,0,-4,0,-6,0,-6,0,-7,0,-8,0,-8,0,-8,0,-9,0,-9,0,-7,0,-5,1,-2,0,-6,0,-9,'rrcurveto',-16,-63,18,'vlineto',0,4,0,7,0,10,0,10,0,10,1,11,1,11,0,10,0,10,0,10,0,7,0,4,'rrcurveto','endchar']
    # table_list = [2,303,67,'rmoveto',-4,-32,-3,-31,'rlineto',1,-6,9,-3,16,0,'rrcurveto',31,1,51,-8,'rlineto',19,7,16,4,13,0,'rrcurveto',12,-1,14,-2,5,3,5,17,'rlineto',1,6,-7,4,-16,3,-16,3,-10,3,-5,3,-5,6,-4,11,-3,16,-9,51,-6,39,-3,27,11,26,5,46,0,67,0,38,-5,26,-9,15,'rrcurveto',-6,10,-21,20,-35,29,'rrcurveto',-62,5,-95,-23,'rlineto',-77,-20,-38,-17,0,-15,0,-19,2,-14,4,-9,'rrcurveto',-10,-12,20,-12,-6,-10,-3,1,'rlineto',4,-1,5,0,7,0,14,0,8,4,1,8,7,27,11,18,16,10,16,10,22,5,29,0,'rrcurveto',51,-34,19,-42,-21,-49,-66,-8,'rlineto',-17,-5,-23,-11,-29,-17,'rrcurveto',-72,-48,'rlineto',-5,-4,-5,-12,-5,-21,-5,-21,-2,-15,0,-9,'rrcurveto',1,-58,'rlineto',15,-15,17,-14,20,-13,24,-15,20,-9,17,-2,'rrcurveto',58,11,50,49,'rlineto',7,6,7,6,8,6,8,6,6,4,4,3,'rrcurveto',-4,93,'rmoveto',-26,-60,-63,-35,-35,4,-18,55,'rlineto',9,27,11,25,13,24,'rrcurveto',46,4,'rlineto',17,13,21,14,25,15,'rrcurveto',-4,-8,1,-22,3,-19,'rlineto','endchar']
    

    if basic:
        table_list = use_basic_operators(table_list, tokenizer)

    if decumulate:
        table_list = make_non_cumulative(table_list, tokenizer, return_string=False)

    if invert:
        table_list = numbers_first(table_list, tokenizer, return_string=False)

    table_list = sort_tablelist(table_list, tokenizer)
    
    viz = Visualizer(table_list)
    viz.draw(filename='./fontmakerai/training_images/ttt.png', plot_outline=True)

    table_list = [2,303,67,'rmoveto',-4,-32,-3,-31,'rlineto',1,-6,9,-3,16,0,'rrcurveto',31,1,51,-8,'rlineto',19,7,16,4,13,0,'rrcurveto',12,-1,14,-2,5,3,5,17,'rlineto',1,6,-7,4,-16,3,-16,3,-10,3,-5,3,-5,6,-4,11,-3,16,-9,51,-6,39,-3,27,11,26,5,46,0,67,0,38,-5,26,-9,15,'rrcurveto',-6,10,-21,20,-35,29,'rrcurveto',-62,5,-95,-23,'rlineto',-77,-20,-38,-17,0,-15,0,-19,2,-14,4,-9,'rrcurveto',-10,-12,20,-12,-6,-10,-3,1,'rlineto',4,-1,5,0,7,0,14,0,8,4,1,8,7,27,11,18,16,10,16,10,22,5,29,0,'rrcurveto',51,-34,19,-42,-21,-49,-66,-8,'rlineto',-17,-5,-23,-11,-29,-17,'rrcurveto',-72,-48,'rlineto',-5,-4,-5,-12,-5,-21,-5,-21,-2,-15,0,-9,'rrcurveto',1,-58,'rlineto',15,-15,17,-14,20,-13,24,-15,20,-9,17,-2,'rrcurveto',58,11,50,49,'rlineto',7,6,7,6,8,6,8,6,6,4,4,3,'rrcurveto',-4,93,'rmoveto',-26,-60,-63,-35,-35,4,-18,55,'rlineto',9,27,11,25,13,24,'rrcurveto',46,4,'rlineto',17,13,21,14,25,15,'rrcurveto',-4,-8,1,-22,3,-19,'rlineto','endchar']

    viz = Visualizer(table_list)
    viz.draw(filename='./fontmakerai/training_images/ttt2.png', plot_outline=True)