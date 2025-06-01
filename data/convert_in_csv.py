import csv

def convert_txt_to_csv(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Text", "Emotion"])  # Header
        for line in infile:
            parts = line.strip().split(";")
            if len(parts) == 2:
                text, label = parts
                writer.writerow([text.strip(), label.strip()])

# Convert each dataset
convert_txt_to_csv("data/train.txt", "train.csv")
convert_txt_to_csv("data/test.txt", "test.csv")
convert_txt_to_csv("data/val.txt", "val.csv")

print("✅ All files converted to CSV successfully!")
