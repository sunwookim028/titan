#!/usr/bin/python3
import csv
import sys

# Function to read a CSV file and return a dictionary of id:data pairs
def read_csv_to_dict(filename):
    with open(filename, mode='r') as file:
        reader = csv.reader(file, delimiter=' ')
        return {row[0]: (row[1:]) for row in reader}


def read_csv_to_dict_ignore_last(filename):
    with open(filename, mode='r') as file:
        reader = csv.reader(file, delimiter=' ')
        return {row[0]: (row[1:-1]) for row in reader}

def is_diff(data1, data2):
    if data1[0] == '*' and data2[0] == '-1':
        return 0
    return data1 != data2

def grade(name, phase, ignore_last):
    # Read both files
    if ignore_last == 1:
        data1 = read_csv_to_dict_ignore_last(f"../bwa-mem2/{name}.{phase}")
        data2 = read_csv_to_dict_ignore_last(f"{name}.{phase}")
    else:
        data1 = read_csv_to_dict(f"../bwa-mem2/{name}.{phase}")
        data2 = read_csv_to_dict(f"{name}.{phase}")

    # Compare the data
    differences = {id: (data1[id], data2[id]) for id in data1 if id in data2 and is_diff(data1[id], data2[id])}

    # Print differences
    prev_id = -1
    count = 0
    for id, (val1, val2) in differences.items():
        print(f"ID: {id}, Correct: {val1}, Out: {val2}")

    print(f"{name} {phase} {(1 - len(differences) / 100000) * 100:.1f}%\n")


    with open('acc', 'a') as file:
        if ignore_last == 1:
            file.write(f"{name} {phase} {(1 - len(differences) / 100000) * 100:.1f}% (last col. ignored)\n")
        else:
            file.write(f"{name} {phase} {(1 - len(differences) / 100000) * 100:.1f}%\n")


def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) < 3:
        print("Usage: python script.py {ecoli|76bp|100bp|152bp} {chchain|...}")
        sys.exit(1)

    name=sys.argv[1]
    phase=sys.argv[2]
    if len(sys.argv) >= 4:
        grade(name, phase, 1) # ignore the last column
    else:
        grade(name, phase, 0)

if __name__ == "__main__":
    main()
