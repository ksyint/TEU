import os

def merge_python_files(output_file="merged_code.txt"):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.py') and file != os.path.basename(__file__):
                    file_path = os.path.join(root, file)
                    outfile.write(f"\n{'='*50}\n")
                    outfile.write(f"File: {file_path}\n")
                    outfile.write(f"{'='*50}\n\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}")
                    outfile.write("\n")

if __name__ == "__main__":
    merge_python_files()