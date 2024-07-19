#!/usr/bin/env python3
import fontforge
import psMat
import os
import sys
import xml.etree.ElementTree as ET

INPUT_SVG = "/Users/alexkoong/Desktop/fonts/letter-a-text-variant-svgrepo-com.svg"
OUTPUT_OTF = "/Users/alexkoong/Desktop/fonts/output423849.otf"

def verify_svg(svg_path):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        print(f"SVG root tag: {root.tag}")
        print(f"SVG namespace: {root.attrib}")
        for child in root:
            print(f"Child element: {child.tag}")
        return True
    except ET.ParseError as e:
        print(f"Error parsing SVG: {str(e)}")
        return False

def svg_to_otf(svg_path, output_path):
    if not verify_svg(svg_path):
        print("SVG file is invalid or empty. Please check the file.")
        return

    font = fontforge.font()
    font.em = 1000

    # Import SVG as a single glyph
    glyph = font.createChar(65)  # 'A'
    try:
        glyph.importOutlines(svg_path)
        print("Imported SVG as a single glyph")
        
        if glyph.foreground.isEmpty():
            print("Warning: Imported glyph has no contours")
        else:
            print("Successfully imported SVG with contours")
            
            # Scale and center the glyph
            bbox = glyph.boundingBox()
            current_width = bbox[2] - bbox[0]
            current_height = bbox[3] - bbox[1]
            scale = min(font.em / current_width, font.em / current_height) * 0.9
            glyph.transform(psMat.scale(scale))
            glyph.left_side_bearing = glyph.right_side_bearing = int((font.em - glyph.width) / 2)
            glyph.width = font.em

    except Exception as e:
        print(f"Error importing SVG: {str(e)}")
        return

    font.fontname = "SVGFont"
    font.familyname = "SVG Font Family"
    font.fullname = "SVG to OTF Font"

    try:
        font.generate(output_path)
        print(f"OTF file generated: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    except Exception as e:
        print(f"Error generating OTF file: {str(e)}")

    print(f"Number of glyphs in font: {len(list(font.glyphs()))}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_SVG):
        print(f"Error: Input SVG file '{INPUT_SVG}' does not exist.")
        sys.exit(1)
    
    svg_to_otf(INPUT_SVG, OUTPUT_OTF)