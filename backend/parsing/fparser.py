from fontTools.ttLib import TTFont
from fontTools.cffLib import CharStrings, CFFFontSet

# Load the font file
font_path = 'ACharmingFont.cff'  # Replace with the actual path to your font file
font = TTFont(font_path)

# Access the CFF table
cff_table = font['CFF '].cff
cff_font = cff_table[0]

# Get the CharStrings from the CFF font
charstrings = cff_font.CharStrings

# Ensure that 'A' is in the CharStrings
glyph_name = 'D'
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
                print(f'Operator: {item}, Stack: {stack}')
                stack.clear()
            elif isinstance(item, (list, tuple)):
                stack.append(item)
            else:
                print(f'Unknown item type: {type(item)}, Value: {item}')
        print(liss)
   
    # Decode and print the charstring program for 'A'
    decode_charstring(glyph_data)
   
    # Optionally, inspect the raw binary data
    raw_charstring_data = glyph_data.bytecode
    print(f'Raw charstring data (hex): {raw_charstring_data.hex()}')
