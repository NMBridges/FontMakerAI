

import glob
import os
import csv
from fontTools.ttLib import TTFont
from fontTools.cffLib import CharStrings, CFFFontSet
from tqdm import tqdm


import string

incomplete_font = 0


def check_font_characters(font_path):
    def get_char_set():
        # Alphanumeric characters
        chars = set(string.ascii_letters + string.digits)
        # Common punctuation
        chars.update(',./?"":;\\')  # Note: '\\' represents a single backslash
        return chars

    try:
        font = TTFont(font_path)
        chars_to_check = get_char_set()
        
        # Use the 'cmap' table for character to glyph mapping
        best_cmap = font.getBestCmap()
        
        # Check if all required characters are present in the font
        all_chars_present = all(ord(char) in best_cmap for char in chars_to_check)

        return all_chars_present
    except Exception as e:
        print(f"Error processing {font_path}: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    font_path = "/path/to/your/font.otf"
    if os.path.exists(font_path):
        result = check_font_characters(font_path)
        print(f"All required characters present: {result}")
    else:
        print(f"Font file not found: {font_path}")







################






# Define your parsing function
def parse_otf_file(charstrings1, character):


    font = TTFont(charstrings1)
    cff_table = font['CFF '].cff
    cff_font = cff_table[0]
    penis = 0
    list_count = 0
    charstrings = cff_font.CharStrings
    glyph_name = character


    if glyph_name not in charstrings:
        penis = 0
        # print(charstrings1)    
    elif check_font_characters(charstrings1):

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

        

        if glyph_name not in charstrings:
            penis = 0
            # print(charstrings1)
        
        elif not hasattr(cff_font.Private, "Subrs"): 
            # print(charstrings1)
            penis = 8

        else:
            glyph_data = charstrings[glyph_name]

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


     
            return expanded_output_list


    else: 
        incomplete_font + 1







num_index_out_of_range =0
num_CFF_error = 0
num_ttf_otf_error = 0



############# glob functions #################c

# Use glob to find all .otf files in the specified directory


# Modified glob function to search nested folders with debugging
def find_otf_files(root_dir):
    print(f"Starting search in: {root_dir}")
    if not os.path.exists(root_dir):
        print(f"Error: The directory {root_dir} does not exist.")
        return []

    otf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if file.lower().endswith('.otf'):
                otf_files.append(full_path)
    
    print(f"Total OTF files found: {len(otf_files)}")
    return otf_files

# Use the new function to find all .otf files in the specified directory and its subdirectories
root_directory = "/Volumes/5tb_alex_drive/Scraped Fonts/all_otf_fonts"
print(f"Root directory set to: {root_directory}")

otf_files = find_otf_files(root_directory)

if not otf_files:
    print("No OTF files found. Please check the root directory path.")
else:
    # Initialize counters
    num_index_out_of_range = 0
    num_CFF_error = 0
    num_ttf_otf_error = 0

    # Collect all parsed data in a list
    all_parsed_data = []
    
    # Create a progress bar
    for file_path in tqdm(otf_files, desc="Processing OTF files", unit="file"):
        try:
            parsed_data = parse_otf_file(file_path, 'a')
            if parsed_data:
                all_parsed_data.append(parsed_data)
        except Exception as e:
            if str(e) == "list index out of range": 
                num_index_out_of_range += 1
            elif str(e) == "not a TrueType or OpenType font (bad sfntVersion)": 
                num_ttf_otf_error += 1
            elif str(e) == "'CFF '": 
                num_CFF_error += 1
                breakpoint()

    print(f"Total files successfully parsed: {len(all_parsed_data)}")
    print(f"Number of CFF errors: {num_CFF_error}")
    print(f"Number of index out of range errors: {num_index_out_of_range}")
    print(f"Number of TTF/OTF errors: {num_ttf_otf_error}")

    # Save all parsed data to a CSV file
    output_csv = "/Volumes/5tb_alex_drive/Scraped Fonts/outputcsv/60k.csv"
    try:
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(all_parsed_data)
        print(f"Data saved to {output_csv}")
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")