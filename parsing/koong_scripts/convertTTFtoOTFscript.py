from fontTools.ttLib import TTFont

def ttf_to_otf(ttf_path, otf_path):
    # Load the TTF font
    font = TTFont(ttf_path)
    
    # Save as OTF font
    font.save(otf_path)

if __name__ == "__main__":
    ttf_path = "/Users/alexkoong/Desktop/Monster AG.ttf"  # Replace with the path to your TTF file
    otf_path = "/Volumes/5tb_alex_drive/Fonts/10000fonts_otf/Monster AG.otf"  # Replace with the desired path for the OTF file
    
    ttf_to_otf(ttf_path, otf_path)
    print(f"Converted {ttf_path} to {otf_path}")
