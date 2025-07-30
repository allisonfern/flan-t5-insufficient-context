import json
import pandas as pd


# preprocess mmau and mmar annotation files:

input_file = "../data/human_ratings_reform_mmau_test_mini.json"
output_csv = "../data/processed_mmaumini_q.csv"

# input_file = "../data/human_ratings_reform_MMAR.json"
# output_csv = "../data/processed_mmar_q.csv"

def determine_label(feedback_list):
    feedback = []
    for f in feedback_list:
        if isinstance(f, str):
            feedback.append(f.lower())
        elif isinstance(f, list):  # flatten nested lists
            feedback.extend(s.lower() for s in f if isinstance(s, str))

    has_q = any("q" in f for f in feedback)
    if has_q:
        return "Question incomplete"
    else:
        return "Sufficient"

all_rows = []
total_q = 0
total_suff = 0

print(f"🔍 Processing {input_file}")
with open(input_file, "r") as f:
    data = json.load(f)

    for _, example in data.items():
        question = example.get("question", "").strip()
        lalms = example.get("lalms", {})

        # aggregate feedback from all models
        all_feedback = []
        for model_data in lalms.values():
            feedback = model_data.get("feedback", [])
            all_feedback.extend(feedback)
        
        label = determine_label(all_feedback)


        if label == "Question incomplete":
            total_q += 1
            all_rows.append({
                "question": question,
                "label": "Question incomplete"
            })
        elif label == "Sufficient":
            total_suff += 1
            all_rows.append({
                "question": question,
                "label": "Sufficient"
            })

# Save to CSV
pd.DataFrame(all_rows).to_csv(output_csv, index=False)

print(f"\n✅ Finished processing.")
print(f"   ➤ Saved {len(all_rows)} examples to {output_csv}")
print("📊 Stats:")
print(f"   • Total examples            : {len(all_rows)}")
print(f"   • Question incomplete (q)   : {total_q}")
print(f"   • Sufficient (neither)      : {total_suff}")
