#!/usr/bin/env python3
# import fontforge
# import psMat
from fontTools import svgLib
from ufo2ft import compileOTF
from defcon import Font
import pathlib
import xml.etree.ElementTree as ET
interest_glyphs = {
    'A': 0x0041, 'B': 0x0042, 'C': 0x0043, 'D': 0x0044, 'E': 0x0045,
    'F': 0x0046, 'G': 0x0047, 'H': 0x0048, 'I': 0x0049, 'J': 0x004A,
    'K': 0x004B, 'L': 0x004C, 'M': 0x004D, 'N': 0x004E, 'O': 0x004F,
    'P': 0x0050, 'Q': 0x0051, 'R': 0x0052, 'S': 0x0053, 'T': 0x0054,
    'U': 0x0055, 'V': 0x0056, 'W': 0x0057, 'X': 0x0058, 'Y': 0x0059,
    'Z': 0x005A, 'a': 0x0061, 'b': 0x0062, 'c': 0x0063, 'd': 0x0064,
    'e': 0x0065, 'f': 0x0066, 'g': 0x0067, 'h': 0x0068, 'i': 0x0069,
    'j': 0x006A, 'k': 0x006B, 'l': 0x006C, 'm': 0x006D, 'n': 0x006E,
    'o': 0x006F, 'p': 0x0070, 'q': 0x0071, 'r': 0x0072, 's': 0x0073,
    't': 0x0074, 'u': 0x0075, 'v': 0x0076, 'w': 0x0077, 'x': 0x0078,
    'y': 0x0079, 'z': 0x007A, '0': 0x0030, '1': 0x0031, '2': 0x0032,
    '3': 0x0033, '4': 0x0034, '5': 0x0035, '6': 0x0036, '7': 0x0037,
    '8': 0x0038, '9': 0x0039, ',': 0x002C, '.': 0x002E, '/': 0x002F,
    '?': 0x003F, '"': 0x0022, ':': 0x003A, ';': 0x003B, '\'': 0x005C
}


def verify_svg(svg_path : pathlib.Path) -> bool:
    '''
    Verifies that an SVG file at a given path is valid.

    Parameters:
    -----------
    svg_path (pathlib.Path): the path to the SVG file

    Returns:
    --------
    bool: whether or not the SVG file was valid
    '''
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        o1 = root.tag
        o2 = root.attrib
        # print(f"SVG root tag: {root.tag}")
        # print(f"SVG namespace: {root.attrib}")
        for child in root:
            o3 = child.tag
            # print(f"Child element: {child.tag}")
        return True
    except ET.ParseError as e:
        print(f"Error parsing SVG: {str(e)}")
        return False


def svgs_to_otf(svg_directory: pathlib.Path, output_path : pathlib.Path):
    '''
    Turns SVGs within a specified directory (corresponding to the unicode
    values of the glyphs specified in config.py) into a font file and saves
    it at the specified location.

    Parameters:
    -----------
    svg_directory (pathlib.Path): a directory containing SVGs with names
                                corresponding to the unicode index of the
                                character
    '''
    font = Font()
    font.em = 1000

    for char, idx in interest_glyphs.items():
        svg_path = svg_directory.joinpath(f'{idx}.svg')
        if not verify_svg(svg_path):
            print(f"SVG file at {svg_path} is invalid or empty. Please check the file.")
            return

        glyph = font.newGlyph(char)
        pen = glyph.getPen()
        path = svgLib.path.SVGPath(svg_path)
        try:
            path.draw(pen)
        except Exception as e:
            print(f"Error importing SVG: {str(e)}")
            return

    otf = compileOTF(font)
    otf.save(output_path)


if __name__ == "__main__":
    dir = pathlib.Path("./svg_test")
    out = pathlib.Path("./svg_test/fonntt.otf")
    svgs_to_otf(dir, out)