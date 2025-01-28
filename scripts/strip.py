# Specify the input and output file paths
import sys
input_file = sys.argv[1]
output_file = sys.argv[2]

# Open the input file and read lines
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Split the line into columns
        columns = line.strip().split()
        
        # Ensure the line has at least 6 columns
        if len(columns) >= 6:
            # Collect the 1st, 4th, and 6th columns
            selected_columns = [columns[1], columns[4], columns[5]]
            
            # Write the selected columns to the output file
            outfile.write("\t".join(selected_columns) + "\n")

