import glob
import os
from fontTools.ttLib import TTFont
from collections import Counter
from tqdm import tqdm
import string

def get_char_set():
    # Alphanumeric characters
    chars = set(string.ascii_letters + string.digits)
    # Common punctuation
    chars.update([',', '.', '/', '?', '"', '"', ':', ';', '\\'])
    return chars

def count_characters(font_path):
    try:
        font = TTFont(font_path)
        chars_to_check = get_char_set()
        
        # Use the 'cmap' table for character to glyph mapping
        best_cmap = font.getBestCmap()
        
        # Count how many of these characters exist in the font
        glyph_count = sum(1 for char in chars_to_check if ord(char) in best_cmap)

        return glyph_count, len(chars_to_check)
    except Exception as e:
        print(f"Error processing {font_path}: {str(e)}")
        return 0, len(get_char_set())

def find_otf_files(base_path):
    pattern = os.path.join(base_path, "**", "*.otf")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} OTF files using pattern: {pattern}")
    if len(files) == 0:
        print("Directory contents:")
        for root, dirs, files in os.walk(base_path):
            level = root.replace(base_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")
    return files

def main():
    # Specify the base directory containing OTF files
    base_directory = "/Volumes/5tb_alex_drive/Scraped Fonts/all_otf_fonts"

    # Check if the directory exists
    if not os.path.exists(base_directory):
        print(f"Error: The specified directory does not exist: {base_directory}")
        return

    # Find all .otf files in the base directory and its subdirectories
    otf_files = find_otf_files(base_directory)

    # Count characters for each font with a progress bar
    glyph_counts = []
    total_chars = len(get_char_set())
    for file_path in tqdm(otf_files, desc="Processing OTF files", unit="file"):
        count, _ = count_characters(file_path)
        if count > 0:
            glyph_counts.append(count)

    # Find the top 10 most common glyph counts
    counter = Counter(glyph_counts)
    top_10 = counter.most_common(10)

    # Print results
    print(f"\nSuccessfully processed {len(glyph_counts)} OTF files.")
    print(f"Total characters checked for each font: {total_chars}")
    print("Top 10 most common glyph counts:")
    for count, frequency in top_10:
        print(f"{count}/{total_chars} glyphs: {frequency} fonts")

if __name__ == "__main__":
    main()