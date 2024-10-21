import os
import lzma
from tqdm import tqdm
from multiprocessing import cpu_count, get_start_method, set_start_method, Lock
import concurrent.futures

# Create a global lock for writing to files
lock = Lock()

# Function to process individual files and write their content
def process_file(args):
    directory, filename, output_file, vocab = args
    file_path = os.path.join(directory, filename)

    # Read from the compressed .xz file
    with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        text = infile.read()

    characters = set(text)

    # Thread-safe writing to the output file
    with lock:
        with open(output_file, "a", encoding="utf-8") as outfile:
            outfile.write(text)

    return characters


# Get all the .xz files in a directory
def xz_files_in_dir(directory):
    return [filename for filename in os.listdir(directory) if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename))]


# Process files in parallel
def process_files_in_parallel(files, folder_path, output_file):
    vocab = set()

    # Using ProcessPoolExecutor for parallel file processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        args = [(folder_path, filename, output_file, vocab) for filename in files]
        for characters in tqdm(executor.map(process_file, args), total=len(files)):
            vocab.update(characters)

    return vocab


if __name__ == "__main__":  # Add this block for compatibility with macOS multiprocessing
    # Set the spawn method for macOS compatibility
    if get_start_method() != 'spawn':
        set_start_method('spawn')

    folder_path = "openwebtext"
    output_file_train = "output_train.txt"
    output_file_val = "output_val.txt"
    vocab_file = "vocab.txt"

    # Get all .xz files
    files = xz_files_in_dir(folder_path)
    total_files = len(files)

    # Split the files for training and validation (90% train, 10% val)
    split_index = int(total_files * 0.9)
    files_train = files[:split_index]
    files_val = files[split_index:]

    # Clear any existing content in the output files
    open(output_file_train, 'w').close()
    open(output_file_val, 'w').close()

    # Process the training files
    print("Processing training files...")
    vocab_train = process_files_in_parallel(files_train, folder_path, output_file_train)

    # Process the validation files
    print("Processing validation files...")
    vocab_val = process_files_in_parallel(files_val, folder_path, output_file_val)

    # Combine vocabularies and write to vocab.txt
    vocab = vocab_train.union(vocab_val)
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in sorted(vocab):
            vfile.write(char + '\n')

    print("Processing complete. Vocabulary saved.")
