import json
import pandas as pd


# preprocess mmau and mmar annotation files:

input_file = "../data/human_ratings_reform_mmau_test_mini.json"
output_csv = "../data/processed_mmaumini_a.csv"

# input_file = "../data/human_ratings_reform_MMAR.json"
# output_csv = "../data/processed_mmar_a.csv"

def determine_label(feedback_list):
    feedback = []
    for f in feedback_list:
        if isinstance(f, str):
            feedback.append(f.lower())
        elif isinstance(f, list):  # flatten nested lists
            feedback.extend(s.lower() for s in f if isinstance(s, str))

    has_a = any("a" in f for f in feedback)
    if has_a:
        return "Rationale insufficient"
    else:
        return "Sufficient"

all_rows = []
total_a = 0
total_suff = 0

print(f"🔍 Processing {input_file}")
with open(input_file, "r") as f:
    data = json.load(f)

    for _, example in data.items():
        question = example.get("question", "").strip()
        rationale = example.get("rationale", "").strip()
        reference = example.get("reference_answer", "").strip()
        
        lalms = example.get("lalms", {})
        candidates = [] # store all candidate answers, to be added if all sufficient
        has_a_annot = False
        for model_name, model_data in lalms.items():
            candidate = model_data.get("candidate", "").strip()
            candidates.append(candidate)
            feedback = model_data.get("feedback", [])
            label = determine_label(feedback)

            if label == "Rationale insufficient":
                has_a_annot = True
                total_a += 1
                all_rows.append({
                    "question": question,
                    "rationale": rationale,
                    "reference": reference,
                    "candidate": candidate,
                    "label": "Rationale insufficient"
                })
        if has_a_annot == False:
            for c in candidates:
                total_suff += 1
                all_rows.append({
                    "question": question,
                    "rationale": rationale,
                    "reference": reference,
                    "candidate": c,
                    "label": "Sufficient"
                })

# Save to CSV
pd.DataFrame(all_rows).to_csv(output_csv, index=False)

print(f"\n✅ Finished processing.")
print(f"   ➤ Saved {len(all_rows)} examples to {output_csv}")
print("📊 Stats:")
print(f"   • Total examples            : {len(all_rows)}")
print(f"   • Rationale insufficient (a): {total_a}")
print(f"   • Sufficient (neither)      : {total_suff}")
