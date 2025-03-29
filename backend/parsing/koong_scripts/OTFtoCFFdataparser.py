from fontTools.ttLib import TTFont
from fontTools.cffLib import CharStrings, CFFFontSet

# Load the font file
font_path = '/Volumes/5tb_alex_drive/Scraped Fonts/all_otf_fonts/output_otf_files/fireworks-kid/Fireworks Kid.otf'  # Replace with the actual path to your font file
font = TTFont(font_path)

# Access the CFF table
cff_table = font['CFF '].cff


cff_font = cff_table[0]

# Get the CharStrings from the CFF font
charstrings = cff_font.CharStrings

# Ensure that 'A' is in the CharStrings
glyph_name = 'a'

if glyph_name not in charstrings:
    print(f"Glyph '{glyph_name}' not found in the CharStrings.")
else:
    glyph_data = charstrings[glyph_name]
   
    # Function to decode and print the charstring program
    def decode_charstring(charstring):
        stack = []
        charstring.decompile()
        liss = []
        for item in charstring.program:
            liss.append(item)
            if isinstance(item, int):
                stack.append(item)
            elif isinstance(item, str):
                # print(f'Operator: {item}, Stack: {stack}')
                stack.clear()
            elif isinstance(item, (list, tuple)):
                stack.append(item)
            else:
                # print(f'Unknown item type: {type(item)}, Value: {item}')
                x=0

            # print(liss)
            # if a new rounding algorithm is needed, here is where you edit it

        rounded_liss = [round(item) if isinstance(item, (int, float)) else item for item in liss]
        # print(rounded_liss)
        return rounded_liss

   
full_table_list = decode_charstring(glyph_data)
   

# print(full_table_list)


############## ACCOUNT FOR SUBR #######


def round_list(input_list):
    output_list = []
    for item in input_list:
        if isinstance(item, float):
            output_list.append(round(item))
        else:
            output_list.append(item)
    return output_list

try:
    local_subrs = cff_font.Private.Subrs
    # print(local_subrs)

    subr_length=0
    subr_array=[]
    bias = 0
    for subr in local_subrs: 
        subr_length+= 1
        subr.decompile()
        table_list = []
        for pen in subr.program: 
            table_list.append(pen)


        subr_array.append(table_list)

    #bias calculation, bias must be ADDED
    if subr_length < 1240: 
        bias = 107
    elif subr_length < 33900: 
        bias = 1131
    else: 
        bias = 32768
except:
    print('no call subr')
# print(round_list(subr_array))
# print(full_table_list)
# print(subr_length)
# print(bias)

dummy_var = 0

expanded_output_list = []

for item_index, item in enumerate(full_table_list): 
    if type(item) == int: 
        expanded_output_list.append(item)
    elif type(item) == str:
        if item == 'callsubr': 
            #what happens if the variable is callsubr
            # expanded_output_list.pop(item_index - 1)
            current_index = len(expanded_output_list) - 1

            unbiased_index = full_table_list[item_index - 1] + bias

            for item in round_list(subr_array[unbiased_index]): 
                expanded_output_list.append(item)
        
            expanded_output_list.pop(current_index)

        else: 
            expanded_output_list.append(item)


print(expanded_output_list)


# pop at: 4, 
