import glob
import os
import csv
from fontTools.ttLib import TTFont
from fontTools.cffLib import CharStrings, CFFFontSet
from tqdm import tqdm
import string

def check_font_characters(font_path):
    def get_char_set():
        chars = set(string.ascii_letters + string.digits)
        chars.update(',./?"":;\\')
        return chars

    try:
        font = TTFont(font_path)
        chars_to_check = get_char_set()
        best_cmap = font.getBestCmap()
        all_chars_present = all(ord(char) in best_cmap for char in chars_to_check)
        return all_chars_present
    except Exception as e:
        print(f"Error processing {font_path}: {str(e)}")
        return False

def parse_otf_file(charstrings1, character):
    try:
        font = TTFont(charstrings1)
        cff_table = font['CFF '].cff
        cff_font = cff_table[0]
        charstrings = cff_font.CharStrings
        glyph_name = character

        if glyph_name not in charstrings or not check_font_characters(charstrings1):
            return None

        if not hasattr(cff_font.Private, "Subrs"):
            return None

        def decode_charstring(charstring):
            charstring.decompile()
            liss = []
            for item in charstring.program:
                liss.append(item)
            rounded_liss = [round(item) if isinstance(item, (int, float)) else item for item in liss]
            return rounded_liss

        glyph_data = charstrings[glyph_name]
        full_table_list = decode_charstring(glyph_data)

        local_subrs = cff_font.Private.Subrs
        subr_length = len(local_subrs)
        subr_array = []
        for subr in local_subrs:
            subr.decompile()
            table_list = []
            for pen in subr.program:
                table_list.append(pen)
            subr_array.append(table_list)

        bias = 107 if subr_length < 1240 else 1131 if subr_length < 33900 else 32768

        expanded_output_list = []
        for item_index, item in enumerate(full_table_list):
            if isinstance(item, int):
                expanded_output_list.append(item)
            elif isinstance(item, str):
                if item == 'callsubr':
                    current_index = len(expanded_output_list) - 1
                    unbiased_index = full_table_list[item_index - 1] + bias
                    for subr_item in [round(i) if isinstance(i, (int, float)) else i for i in subr_array[unbiased_index]]:
                        expanded_output_list.append(subr_item)
                    expanded_output_list.pop(current_index)
                else:
                    expanded_output_list.append(item)

        return expanded_output_list
    except Exception as e:
        print(f"Error parsing {charstrings1}: {str(e)}")
        return None

def find_otf_files(root_dir):
    print(f"Starting search in: {root_dir}")
    if not os.path.exists(root_dir):
        print(f"Error: The directory {root_dir} does not exist.")
        return []

    otf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.otf'):
                otf_files.append(os.path.join(root, file))
    
    print(f"Total OTF files found: {len(otf_files)}")
    return otf_files

def process_otf_files(root_directory, temp_csv_path):
    otf_files = find_otf_files(root_directory)

    if not otf_files:
        print("No OTF files found. Please check the root directory path.")
        return

    num_index_out_of_range = 0
    num_CFF_error = 0
    num_ttf_otf_error = 0

    all_parsed_data = []
    
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

    print(f"Total files successfully parsed: {len(all_parsed_data)}")
    print(f"Number of CFF errors: {num_CFF_error}")
    print(f"Number of index out of range errors: {num_index_out_of_range}")
    print(f"Number of TTF/OTF errors: {num_ttf_otf_error}")

    with open(temp_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_parsed_data)
    print(f"Raw data saved to {temp_csv_path}")

def deduplicate_csv(input_file, output_file):
    unique_rows = set()
    
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            if any("hint" in cell.lower() or "stem" in cell.lower() for cell in row):
                continue
            
            cleaned_row = ','.join(row).replace('return', '').split(',')
            cleaned_row = [cell for cell in cleaned_row if cell]
            
            row_tuple = tuple(cleaned_row)
            if row_tuple not in unique_rows:
                unique_rows.add(row_tuple)
                writer.writerow(cleaned_row)

    print(f"Deduplicated and cleaned CSV saved to {output_file}")

if __name__ == "__main__":
    root_directory = "/Volumes/5tb_alex_drive/Scraped Fonts/all_otf_fonts"
    temp_csv_path = "/Volumes/5tb_alex_drive/Scraped Fonts/outputcsv/temp_raw_output.csv"
    final_output_csv = "/Volumes/5tb_alex_drive/Scraped Fonts/outputcsv/final_cleaned_output.csv"

    process_otf_files(root_directory, temp_csv_path)
    deduplicate_csv(temp_csv_path, final_output_csv)

    # Optionally, remove the temporary CSV file
    # os.remove(temp_csv_path)
