import re

def extract_pos_and_words(treebank_str):

    # use regex to fit word format
    pattern = r'\((\S+)\s+(\S+)\)'

    matches = re.findall(pattern, treebank_str)
    pos_word_pairs = [f"{pos} {word}" for pos, word in matches if pos != "-NONE-"] 

    return pos_word_pairs

def process_file(input_file, output_file):

    with open(input_file, 'r') as infile:
        content = infile.read()

    pos_word_pairs = extract_pos_and_words(content)

    with open(output_file, 'w') as outfile:
        outfile.write("\n".join(pos_word_pairs))

input_file = "output13.mrg"
output_file = "extracted13.txt"

process_file(input_file, output_file)

