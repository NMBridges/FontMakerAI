
# import csv
# from collections import Counter
# import matplotlib.pyplot as plt

# def count_and_graph_top_values(file_path, top_n=100):
#     try:
#         # Initialize a counter for all values
#         value_counter = Counter()

#         # Open and read the CSV file
#         with open(file_path, 'r', newline='', encoding='utf-8') as file:
#             csv_reader = csv.reader(file)
            
#             # Skip the header row
#             next(csv_reader, None)
            
#             # Count all values in the file
#             for row in csv_reader:
#                 value_counter.update(row)

#         # Get the top N most common values
#         top_values = value_counter.most_common(top_n)

#         print(f"The {top_n} most common values in the entire CSV file:")
#         for value, count in top_values:
#             print(f"'{value}': {count} times")

#         # Prepare data for plotting
#         values, counts = zip(*top_values)

#         # Create a bar plot
#         plt.figure(figsize=(15, 10))
#         plt.bar(range(len(values)), counts)
#         plt.title(f"Top {top_n} Most Common Values in CSV File")
#         plt.xlabel("Values")
#         plt.ylabel("Frequency")
#         plt.xticks(range(len(values)), [str(v)[:20] for v in values], rotation=90, ha='right')
#         plt.tight_layout()

#         # Save the plot
#         plt.savefig('top_values_distribution.png')
#         print("\nA bar plot of the top values has been saved as 'top_values_distribution.png'")

#         # Show the plot (optional, comment out if running in a non-interactive environment)
#         plt.show()

#         return top_values

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         return None

# # Example usage
# file_path = '/Users/alexkoong/Desktop/46889.csv'
# count_and_graph_top_values(file_path)

import csv
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_adjacent_cooccurrence_matrix(file_path, top_n=50):
    try:
        # Initialize a counter for all values
        value_counter = Counter()

        # First pass: Count all values to find top N
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                value_counter.update(row)

        # Get top N tokens
        top_tokens = [token for token, _ in value_counter.most_common(top_n)]
        token_to_index = {token: i for i, token in enumerate(top_tokens)}
        
        # Initialize co-occurrence matrix
        cooccurrence_matrix = np.zeros((top_n, top_n), dtype=int)
        
        # Second pass: Build co-occurrence matrix
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                for i in range(len(row)):
                    if row[i] in token_to_index:
                        current_index = token_to_index[row[i]]
                        # Count self-occurrence
                        cooccurrence_matrix[current_index][current_index] += 1
                        
                        # Check next token if it exists
                        if i + 1 < len(row) and row[i+1] in token_to_index:
                            next_index = token_to_index[row[i+1]]
                            cooccurrence_matrix[current_index][next_index] += 1
                            cooccurrence_matrix[next_index][current_index] += 1

        # Create a DataFrame for better display
        df = pd.DataFrame(cooccurrence_matrix, index=top_tokens, columns=top_tokens)
        
        # Save the matrix to a CSV file
        df.to_csv('adjacent_cooccurrence_matrix.csv')
        print("Adjacent co-occurrence matrix saved as 'adjacent_cooccurrence_matrix.csv'")

        # Create a heatmap
        plt.figure(figsize=(15, 13))
        sns.heatmap(df, cmap='YlOrRd', annot=False, square=True)
        plt.title('Adjacent Co-occurrence Matrix Heatmap')
        plt.tight_layout()
        plt.savefig('adjacent_cooccurrence_heatmap.png', dpi=300, bbox_inches='tight')
        print("Heatmap saved as 'adjacent_cooccurrence_heatmap.png'")

        return df, top_tokens

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

# Example usage
file_path = '/Users/alexkoong/Desktop/46889.csv'
matrix_df, tokens = create_adjacent_cooccurrence_matrix(file_path)

if matrix_df is not None and tokens is not None:
    print("\nTop 50 tokens:")
    for i, token in enumerate(tokens, 1):
        print(f"{i}. {token}")

    print("\nAdjacent co-occurrence matrix (including self-occurrences):")
    print(matrix_df)

    # Display a portion of the matrix (e.g., top-left 10x10) for quick view
    print("\nSample of adjacent co-occurrence matrix (top-left 10x10):")
    print(matrix_df.iloc[:10, :10])

    print("\nFull matrix has been saved to 'adjacent_cooccurrence_matrix.csv'")
    print("Heatmap has been saved to 'adjacent_cooccurrence_heatmap.png'")