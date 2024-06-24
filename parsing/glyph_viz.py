import numpy as np
import matplotlib.pyplot as plt


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
                elif len(numbers) == 3 and running_idx == 4: # must be first operator in sequence
                    width = numbers[0]
                    cX += numbers[1]
                    cY += numbers[2]
                    paths.new_path(cX, cY)
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
                elif len(numbers) % rep_size == 1 and len(numbers) > 0:
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
                elif len(numbers) % rep_size == 1 and len(numbers) > 0:
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
                if len(numbers) % rep_size == 2 and len(numbers) > 2:
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
                # Not in any of the fonts
                pass
            elif operator == "vstem":
                # Not in any of the fonts
                pass
            elif operator == "hstemhm":
                # Not in any of the fonts
                pass
            elif operator == "vstemhm":
                # Not in any of the fonts
                pass
            elif operator == "hintmask":
                # Not in any of the fonts
                pass
            elif operator == "cntrmask":
                # Not in any of the fonts
                pass
            elif operator == "callsubr":
                pass
            elif operator == "callgsubr":
                pass
            elif operator == "vsindex":
                # Not in any of the fonts
                pass
            elif operator == "blend":
                # Not in any of the fonts
                pass

            elif operator == "endchar":
                break
            
            else:
                raise Exception("Cannot end table list without an operator (specifically, an endchar")
            
        return paths.get_paths()

    def draw(self, display : bool = True, filename : str = None):
        paths = self.get_paths()

        for path in paths:
            if path[0] != path[-1]:
                path.append(path[0])
            plt.plot(*zip(*path))
            plt.scatter(*zip(*path))
        plt.gca().set_aspect('equal')
        
        if filename:
            plt.savefig(filename)
        if display:
            plt.show()
        # TODO: matplotlib stuff


if __name__ == "__main__":
    table_list = [2,303,67,'rmoveto',-4,-32,-3,-31,'rlineto',1,-6,9,-3,16,0,'rrcurveto',31,1,51,-8,'rlineto',19,7,16,4,13,0,'rrcurveto',12,-1,14,-2,5,3,5,17,'rlineto',1,6,-7,4,-16,3,-16,3,-10,3,-5,3,-5,6,-4,11,-3,16,-9,51,-6,39,-3,27,11,26,5,46,0,67,0,38,-5,26,-9,15,'rrcurveto',-6,10,-21,20,-35,29,'rrcurveto',-62,5,-95,-23,'rlineto',-77,-20,-38,-17,0,-15,0,-19,2,-14,4,-9,'rrcurveto',-10,-12,20,-12,-6,-10,-3,1,'rlineto',4,-1,5,0,7,0,14,0,8,4,1,8,7,27,11,18,16,10,16,10,22,5,29,0,'rrcurveto',51,-34,19,-42,-21,-49,-66,-8,'rlineto',-17,-5,-23,-11,-29,-17,'rrcurveto',-72,-48,'rlineto',-5,-4,-5,-12,-5,-21,-5,-21,-2,-15,0,-9,'rrcurveto',1,-58,'rlineto',15,-15,17,-14,20,-13,24,-15,20,-9,17,-2,'rrcurveto',58,11,50,49,'rlineto',7,6,7,6,8,6,8,6,6,4,4,3,'rrcurveto',-4,93,'rmoveto',-26,-60,-63,-35,-35,4,-18,55,'rlineto',9,27,11,25,13,24,'rrcurveto',46,4,'rlineto',17,13,21,14,25,15,'rrcurveto',-4,-8,1,-22,3,-19,'rlineto','endchar']
    viz = Visualizer(table_list)
    viz.draw(filename='ttt.png')