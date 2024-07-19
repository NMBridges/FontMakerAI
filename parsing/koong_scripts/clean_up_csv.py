import csv
import os

# Set your input and output file paths here
INPUT_FILE = r"/Volumes/5tb_alex_drive/Scraped Fonts/outputcsv/60k.csv"
OUTPUT_FILE = r"/Volumes/5tb_alex_drive/Scraped Fonts/outputcsv/60kcleaned.csv"

def deduplicate_csv(input_file, output_file):
    # Set to store unique rows
    unique_rows = set()
    
    # Read input file and write to output file
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        header = next(reader)
        writer.writerow(header)
        
        # Process rows
        for row in reader:
            # Check if the row contains "hint" or "stem"
            if any("hint" in cell.lower() or "stem" in cell.lower() for cell in row):
                continue  # Skip this row
            
            # Clean the row using the suggested method
            cleaned_row = ','.join(row).replace('return', '').split(',')
            
            # Remove any empty strings that might have been created
            cleaned_row = [cell for cell in cleaned_row if cell]
            
            row_tuple = tuple(cleaned_row)  # Convert row to tuple for hashing
            if row_tuple not in unique_rows:
                unique_rows.add(row_tuple)
                writer.writerow(cleaned_row)

    print(f"Deduplicated and cleaned CSV saved to {output_file}")

if __name__ == "__main__":
    # Check if input file exists
    if not os.path.isfile(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' does not exist.")
    else:
        deduplicate_csv(INPUT_FILE, OUTPUT_FILE)