from fontTools.ttLib import TTFont
from fontTools.cffLib import CharStrings

# Function to compile and set the charstring
def set_charstring(font, glyph_name, charstring_data):
    cff_table = font['CFF '].cff
    cff_font = cff_table[0]
    charstrings = cff_font.CharStrings

    if glyph_name not in charstrings:
        print(f"Glyph '{glyph_name}' not found in the CharStrings.")
        return

    charstring = charstrings[glyph_name]
    charstring.program = []

    for item in charstring_data:
        if isinstance(item, (int, float)):
            charstring.program.append(item)
        elif isinstance(item, str):
            charstring.program.append(item)
        else:
            print(f"Unknown item type: {type(item)}, Value: {item}")

    charstring.bytecode = charstring.compile()
    charstrings[glyph_name] = charstring

# Load the original OTF file
font_path = '/Users/alexkoong/Desktop/acharmingfont.otf'  # Replace with the actual path to your font file
font = TTFont(font_path)

# CharString data to be set
charstring_data =  []


# Glyph name to update
glyph_name = 'a'

# Set the new charstring data
set_charstring(font, glyph_name, charstring_data)

# Save the modified font
output_font_path = '/Users/alexkoong/Desktop/updatedfont.otf'  # Replace with the desired output path
font.save(output_font_path)

print(f"Updated font saved to {output_font_path}")






