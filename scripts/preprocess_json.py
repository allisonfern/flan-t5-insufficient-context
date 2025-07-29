import json
import pandas as pd


# preprocess mmau and mmar annotation files:

# input_file = "data/reformatted/human_ratings_reform_mmau_test_mini.json"
# output_csv = "data/reformatted/reform_processed_mmaumini.csv"

input_file = "data/reformatted/human_ratings_reform_MMAR.json"
output_csv = "data/reformatted/reform_processed_mmar.csv"

def determine_label(feedback_list):
    feedback = []
    for f in feedback_list:
        if isinstance(f, str):
            feedback.append(f.lower())
        elif isinstance(f, list):  # flatten nested lists
            feedback.extend(s.lower() for s in f if isinstance(s, str))

    has_q = any("q" in f for f in feedback)
    has_a = any("a" in f for f in feedback)
    # how to handle instances with both annotations?
    if has_q and has_a:
        return "Both"
    if has_q:
        return "Question incomplete"
    if has_a:
        return "Rationale insufficient"
    return "Sufficient"

all_rows = []
total_q = 0
total_a = 0
total_both = 0

print(f"🔍 Processing {input_file}")
with open(input_file, "r") as f:
    data = json.load(f)

    for _, example in data.items():
        question = example.get("question", "").strip()
        rationale = example.get("rationale", "").strip()
        reference = example.get("reference_answer", "").strip()
        lalms = example.get("lalms", {})

        # Track if any model has feedback
        selected_row = None
        candidates = []
        gen_label = "Sufficient" # keep track of the label to apply to all instances
        for model_name, model_data in lalms.items():
            candidate = model_data.get("candidate", "").strip()
            feedback = model_data.get("feedback", [])
            
            candidates.append(candidate) # store all candidate answers in a list

            label = determine_label(feedback)

            if label == "Both":
                total_both +=1
            elif label == "Question incomplete":
                total_q += 1
                gen_label = label
            elif label == "Rationale insufficient":
                total_a += 1
                gen_label = label


        if label == "Both":
            all_rows.append({
                "question": question,
                "rationale": rationale,
                "reference": reference,
                "candidates": candidates,
                "label": "Question incomplete"
            })
            all_rows.append({
                "question": question,
                "rationale": rationale,
                "reference": reference,
                "candidates": candidates,
                "label": "Rationale insufficient"
            })

        else:
            selected_row = {
                "question": question,
                "rationale": rationale,
                "reference": reference,
                "candidates": candidates,
                "label": gen_label
            }

            all_rows.append(selected_row)

# Save to CSV
pd.DataFrame(all_rows).to_csv(output_csv, index=False)

print(f"\n✅ Finished processing.")
print(f"   ➤ Saved {len(all_rows)} examples to {output_csv}")
print("📊 Stats:")
print(f"   • Total examples            : {len(all_rows)}")
print(f"   • Question incomplete (q)   : {total_q}")
print(f"   • Rationale insufficient (a): {total_a}")
print(f"   • Sufficient (neither)      : {len(all_rows) - total_q - total_a}")
print(f"   • Both                      : {total_both}")
