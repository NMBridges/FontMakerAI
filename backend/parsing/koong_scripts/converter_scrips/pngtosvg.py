import sys
from PIL import Image
import svgwrite
import numpy as np

# User-configurable variables
INPUT_FILE = "/Users/alexkoong/Desktop/fonts/test9.png"  # Change this to your input PNG file path
OUTPUT_FILE = "/Users/alexkoong/Desktop/fonts/OUTPUT9.svg"  # Change this to your desired output SVG file path

def png_to_svg(input_file, output_file):
    # Open the PNG file
    image = Image.open(input_file)

    # Convert to grayscale
    image = image.convert('L')

    # Convert the image to a numpy array
    image_data = np.array(image)

    # Create an SVG drawing
    dwg = svgwrite.Drawing(output_file, size=image.size)

    # Iterate through the image data
    for y in range(image_data.shape[0]):
        for x in range(image_data.shape[1]):
            if image_data[y, x] < 128:  # Adjust this threshold as needed
                dwg.add(dwg.rect(insert=(x, y), size=(1, 1), fill='black'))

    # Save the SVG file
    dwg.save()

    print(f"Conversion complete. SVG file saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # If command-line arguments are provided, use them
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Otherwise, use the variables defined at the top of the script
        input_file = INPUT_FILE
        output_file = OUTPUT_FILE

    png_to_svg(input_file, output_file)