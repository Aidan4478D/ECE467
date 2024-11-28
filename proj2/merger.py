import os

output_file = "output13.mrg"

# Open the output file in write mode
with open(output_file, "w") as outfile:
    for file_name in sorted(os.listdir("./13/")):
        if file_name.endswith(".mrg"):
            print(f"Processing: {file_name}")
            with open("./13/" + file_name, "r") as infile:
                content = infile.read()
                # Write the content to the output file
                outfile.write(content)
                outfile.write("\n")

print(f"All files have been merged into {output_file}")
