import pandas as pd

def process_question_only_txt(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Split blocks by the delimiter line
    blocks = content.strip().split("=" * 100)

    all_rows = []
    for block in blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue

        question = lines[1] # skip id, grab question
        label_line = lines[-1] 

        label = "Question incomplete" if label_line == "1" else "Sufficient"

        all_rows.append({
            "question": question,
            "label": label
        })

    return pd.DataFrame(all_rows)

# Example usage
input_txt = "../data/mmau_question_annot.txt"
output_csv = "../data/processed_q_only.csv"

df = process_question_only_txt(input_txt)
df.to_csv(output_csv, index=False)

print(f"✅ Saved {len(df)} rows to {output_csv}")
