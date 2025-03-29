# import glob
# import os
# from fontTools.ttLib import TTFont
# from fontTools.cffLib import CharStrings, CFFFontSet




# # Define your parsing function
# def parse_otf_file(charstrings1):

#     font = TTFont(charstrings1)
#     cff_table = font['CFF '].cff
#     cff_font = cff_table[0]

#     charstrings = cff_font.CharStrings
#     glyph_name = 'a'

#     if glyph_name not in charstrings:
#         # print(f"Glyph '{glyph_name}' not found in the CharStrings.")
#         # print(charstrings1)
#         x=10
#     else:
#         glyph_data = charstrings[glyph_name]
    
#         # Function to decode and print the charstring program
#         def decode_charstring(charstring):
#             stack = []
#             charstring.decompile()
#             liss = []
#             for item in charstring.program:
#                 liss.append(item)
#                 if isinstance(item, int):
#                     stack.append(item)
#                 elif isinstance(item, str):
#                     # print(f'Operator: {item}, Stack: {stack}')
#                     stack.clear()
#                 elif isinstance(item, (list, tuple)):
#                     stack.append(item)
#                 else:
#                     # print(f'Unknown item type: {type(item)}, Value: {item}')
#                     x=0

#                 # print(liss)
#                 # if a new rounding algorithm is needed, here is where you edit it

#             rounded_liss = [round(item) if isinstance(item, (int, float)) else item for item in liss]
#             print(rounded_liss)
#             return rounded_liss

#         decode_charstring(glyph_data)






# # Use glob to find all .otf files in the specified directory
# otf_files = glob.glob("/Users/alexkoong/fontconvert/testsrc/*.otf")

# # Loop through each .otf file and apply your parsing function
# for file_path in otf_files:
#     parsed_data = parse_otf_file(file_path)
#     # Do something with the parsed data
#     # print(parsed_data)  # Example action

# # Optionally, you can collect all parsed data in a list
# all_parsed_data = []
# for file_path in otf_files:
#     parsed_data = parse_otf_file(file_path)
#     all_parsed_data.append(parsed_data)


# # Now `all_parsed_data` contains parsed data from all files
# # print (all_parsed_data)





############## new version 
import glob
import os
import csv
from fontTools.ttLib import TTFont
from fontTools.cffLib import CharStrings, CFFFontSet



# Define your parsing function
def parse_otf_file(charstrings1):

    font = TTFont(charstrings1)
    cff_table = font['CFF '].cff
    cff_font = cff_table[0]

    list_count = 0
    charstrings = cff_font.CharStrings
    glyph_name = 'a'


    def decode_charstring(charstring):
            stack = []
            charstring.decompile()
            liss = []
            for item in charstring.program:
                liss.append(item)
                if isinstance(item, int):
                    stack.append(item)
                elif isinstance(item, str):
                    stack.clear()
                elif isinstance(item, (list, tuple)):
                    stack.append(item)
                else:
                    x=0

            rounded_liss = [round(item) if isinstance(item, (int, float)) else item for item in liss]

            return rounded_liss


    if glyph_name not in charstrings:
        x=10
    else:
        glyph_data = charstrings[glyph_name]
    

    #############################
        # Function to decode and print the charstring program
# or len(decode_charstring(glyph_data)) < 50 
        if decode_charstring(glyph_data).__contains__("callsubr") : 
            # print(charstrings1)
            list_count += 1
        else:

            if len(decode_charstring(glyph_data)) == 14: 
                print(charstrings1)

            return decode_charstring(glyph_data)





# Use glob to find all .otf files in the specified directory
otf_files = glob.glob("/Users/alexkoong/fontconvert/dst/*.otf")


min_length = 10000 # outside of for loop
min_font = None # outside of for loop


# Collect all parsed data in a list
all_parsed_data = []
for file_path in otf_files:
    parsed_data = parse_otf_file(file_path)
    if parsed_data:
        all_parsed_data.append(parsed_data)

# Save all parsed data to a CSV file
output_csv = "parsed_data_just_filter_subcommand.csv"
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(all_parsed_data)

print(f"Data saved to {output_csv}")

