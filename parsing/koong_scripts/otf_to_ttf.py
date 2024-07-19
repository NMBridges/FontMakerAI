import fontforge
import os
import glob

def convert_otf_to_ttf(input_folder, output_folder):
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_folder):
        # Get all .otf files in the current directory
        otf_files = glob.glob(os.path.join(root, '*.otf'))
        
        for otf_file in otf_files:
            try:
                # Open the font
                font = fontforge.open(otf_file)
                
                # Create relative path
                rel_path = os.path.relpath(root, input_folder)
                
                # Create corresponding output directory
                output_dir = os.path.join(output_folder, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate the output filename
                output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(otf_file))[0] + '.ttf')
                
                # Generate the TTF font
                font.generate(output_file)
                
                # Close the font
                font.close()
                
                print(f"Converted {otf_file} to {output_file}")
            except Exception as e:
                print(f"Error converting {otf_file}: {str(e)}")

# Specify the input folder containing .otf files
input_folder = '/Volumes/5tb_alex_drive/Scraped Fonts/fontmeme/unzipped_fontmeme'

# Specify the output folder for .ttf files
output_folder = '/Volumes/5tb_alex_drive/Scraped Fonts/fontmeme/fontmeme_ttf'

# Run the conversion
convert_otf_to_ttf(input_folder, output_folder)