##ONE FILE TEST

# import fontforge
# import os
# import re

# # Set your input and output file paths here
# INPUT_FILE = r"/Volumes/5tb_alex_drive/Scraped Fonts/all_converted_unconverted_fonts/unzipped_fonts_all/28-days-later/28 Days Later.ttf"
# OUTPUT_FILE = r"/Volumes/5tb_alex_drive/Scraped Fonts/all_converted_unconverted_fonts/unzipped_fonts_all/28-days-later/28 Days Later.otf"

# def convert_ttf_to_otf(input_file, output_file):
#     try:
#         # Open the font
#         font = fontforge.open(input_file)

#         # Set OS/2 version to 5
#         font.os2_version = 4  # This sets the OS/2 table version

#         # Check and modify family name if it starts with a number
#         family_name = font.familyname
#         if family_name[0].isdigit():
#             new_family_name = re.sub(r'^\d+', '', family_name).strip()
#             font.familyname = new_family_name
#             print(f"Family name changed from '{family_name}' to '{new_family_name}'")

#         # Generate OTF file
#         font.generate(output_file)
#         print(f"Converted {input_file} to {output_file}")

#         # Close the font
#         font.close()

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         return False

#     return True

# def main():
#     input_file = os.path.abspath(INPUT_FILE)
#     output_file = os.path.abspath(OUTPUT_FILE)

#     if not input_file.lower().endswith('.ttf'):
#         print("Input file must be a .ttf file")
#         return

#     if not output_file.lower().endswith('.otf'):
#         print("Output file must be an .otf file")
#         return

#     if not os.path.exists(input_file):
#         print(f"Input file {input_file} does not exist")
#         return

#     # Ensure the directory for the output file exists
#     output_dir = os.path.dirname(output_file)
#     if not os.path.exists(output_dir):
#         try:
#             os.makedirs(output_dir)
#             print(f"Created directory: {output_dir}")
#         except OSError as e:
#             print(f"Error creating directory {output_dir}: {e}")
#             return

#     if convert_ttf_to_otf(input_file, output_file):
#         print("Conversion completed successfully.")
#     else:
#         print("Conversion failed.")

# if __name__ == "__main__":
#     main()





##### folder of ttf to otf ####

import fontforge
import os
import re
import glob

# Set your input and output folder paths here
INPUT_FOLDER = r"/Volumes/5tb_alex_drive/Scraped Fonts/fontmeme/unzipped_fontmeme/fontememe_all_ttfs/fontmeme_otf_now_ttf"
OUTPUT_FOLDER = r"/Volumes/5tb_alex_drive/Scraped Fonts/fontmeme/fontmeme_all_otfs"

def convert_ttf_to_otf(input_file, output_file):
    try:
        # Open the font
        font = fontforge.open(input_file)

        # Set OS/2 version to 4
        font.os2_version = 4  # This sets the OS/2 table version

        # Check and modify family name if it starts with a number
        family_name = font.familyname
        if family_name[0].isdigit():
            new_family_name = re.sub(r'^\d+', '', family_name).strip()
            font.familyname = new_family_name
            print(f"Family name changed from '{family_name}' to '{new_family_name}'")

        # Generate OTF file
        font.generate(output_file)
        print(f"Converted {input_file} to {output_file}")

        # Close the font
        font.close()

    except Exception as e:
        print(f"An error occurred with {input_file}: {str(e)}")
        return False

    return True

def main():
    # Ensure the output folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        try:
            os.makedirs(OUTPUT_FOLDER)
            print(f"Created output directory: {OUTPUT_FOLDER}")
        except OSError as e:
            print(f"Error creating directory {OUTPUT_FOLDER}: {e}")
            return

    # Use glob to find all .ttf files in the input folder and its subfolders
    for input_file in glob.glob(os.path.join(INPUT_FOLDER, "**", "*.ttf"), recursive=True):
        # Create the corresponding output file path
        rel_path = os.path.relpath(input_file, INPUT_FOLDER)
        output_file = os.path.join(OUTPUT_FOLDER, os.path.splitext(rel_path)[0] + ".otf")

        # Ensure the output subfolder exists
        output_subfolder = os.path.dirname(output_file)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        if convert_ttf_to_otf(input_file, output_file):
            print(f"Conversion completed successfully for {input_file}")
        else:
            print(f"Conversion failed for {input_file}")

if __name__ == "__main__":
    main()