import pandas as pd
import os


def concatenate_files(file1, file2, output_file, sep='\t', encoding='ISO-8859-1'):
    """
    Concatenate two text files without altering the 'user_id' or other columns.

    :param file1: Path to the first text file.
    :param file2: Path to the second text file.
    :param output_file: Path where the concatenated output will be saved.
    :param sep: Separator used in the text files. Default is tab.
    :param encoding: Encoding of the text files.
    """
    # Read the files into pandas DataFrames with the specified separator, encoding, and no header
    # All columns are read as strings to avoid any unintended data type conversion
    df1 = pd.read_csv(file1, sep=sep, encoding=encoding, header=None, dtype=str)
    df2 = pd.read_csv(file2, sep=sep, encoding=encoding, header=None, dtype=str)

    # Concatenate the DataFrames without sorting or reindexing
    concatenated_df = pd.concat([df1, df2], ignore_index=True)


    # Save the concatenated DataFrame to a new text file
    concatenated_df.to_csv(output_file, sep=sep, index=False, header=False, encoding=encoding)

    print(f"Files have been concatenated and saved to {output_file}")


concatenate_files('processed_nyc.txt', 'augmented_data_new_users.txt', 'output.txt', sep='\t')
