#visualize an OTF file 

from fontTools.ttLib import TTFont

# Open the OTF font
font_path = '/Users/alexkoong/Desktop/acharmingfont.otf'
font = TTFont(font_path)

# Access the CFF table
cff_table = font['CFF ']

print("CFF Table BELOW !!!!")
print(cff_table)

# Print the name of the font
font_name = cff_table.cff.fontNames[0]
print(f"Font Name: {font_name}")

# Access a specific glyph
glyph_set = font.getGlyphSet()
glyph_name = 'A'  # Replace with the glyph name you want to visualize
glyph = glyph_set[glyph_name]



# Print glyph information
print(glyph)

# Draw the glyph using matplotlib
from fontTools.pens.basePen import BasePen
import matplotlib.pyplot as plt

class GlyphDrawer(BasePen):
    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        self.points = []

    def _moveTo(self, p):
        self.points.append([p])

    def _lineTo(self, p):
        self.points[-1].append(p)

    def _curveToOne(self, p1, p2, p3):
        self.points[-1].extend([p1, p2, p3])

    def _closePath(self):
        self.points[-1].append(self.points[-1][0])

    def draw(self, ax):
        for contour in self.points:
            contour = contour + [contour[0]]
            xs, ys = zip(*contour)
            ax.plot(xs, ys)

def draw_glyph(glyph_name, font_path):
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    glyph = glyph_set[glyph_name]

    drawer = GlyphDrawer(glyph_set)
    glyph.draw(drawer)

    fig, ax = plt.subplots()
    drawer.draw(ax)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.show()

print("Gliff Set below")
print(glyph_set)


font_path = '/Users/alexkoong/Desktop/fonts_dataset1/Happy Memories.otf'
glyph_name = 'A'  
draw_glyph(glyph_name, font_path)




#################




# Access the CFF table
cff_table = font['CFF '].cff
cff_font = cff_table[0]

# Get the CharStrings from the CFF font
charstrings = cff_font.CharStrings

# Ensure that 'A' is in the CharStrings
glyph_name = 'A'
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
    #print(f'Raw charstring data (hex): {raw_charstring_data.hex()}')