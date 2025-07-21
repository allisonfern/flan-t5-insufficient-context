import json
import pandas as pd

input_file = "data/human_ratings_july_16.json"
output_csv = "data/processed.csv"

def determine_label(feedback_lists):
    feedback = [item for sublist in feedback_lists if sublist for item in sublist]
    has_q = "q" in feedback
    has_a = "a" in feedback

    if has_q:
        return "Question incomplete", has_q, has_a
    if has_a:
        return "Rationale insufficient", has_q, has_a
    return "Sufficient", has_q, has_a

all_rows = []
total_q = 0
total_a = 0
total_neither = 0

print(f"🔍 Processing {input_file}")
with open(input_file, "r") as f:
    data = json.load(f)

    for _, example in data.items():
        question = example.get("question", "").strip()
        rationale = example.get("rationale", "").strip()
        reference = example.get("reference", "").strip()
        candidate = example.get("candidate", "").strip()
        feedback = example.get("feedback", [])

        if not question or not rationale or not reference or not candidate:
            continue

        label, has_q, has_a = determine_label(feedback)

        if has_q:
            total_q += 1
        elif has_a:
            total_a += 1
        else:
            total_neither += 1

        all_rows.append({
            "question": question,
            "rationale": rationale,
            "reference": reference,
            "candidate": candidate,
            "label": label
        })

pd.DataFrame(all_rows).to_csv(output_csv, index=False)

print(f"\n✅ Finished processing.")
print(f"   ➤ Saved {len(all_rows)} examples to {output_csv}")
print("📊 Stats:")
print(f"   • Total questions           : {len(all_rows)}")
print(f"   • Question incomplete (q)   : {total_q}")
print(f"   • Rationale insufficient (a): {total_a}")
print(f"   • Sufficient (neither)      : {total_neither}")
