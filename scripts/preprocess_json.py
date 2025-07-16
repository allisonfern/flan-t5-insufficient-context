import json
import re
import pandas as pd
from bs4 import BeautifulSoup
import os
import glob

INPUT_PATTERN = "data/*.jsonl"

def extract_from_html(html):
    soup = BeautifulSoup(html, "html.parser")

    def get_field_by_strong(label):
        tag = soup.find("strong", string=re.compile(label))
        if tag:
            text_parts = []
            for sibling in tag.next_siblings:
                if sibling.name == "strong" or (sibling.name == "br" and len(text_parts) > 0):
                    break
                if isinstance(sibling, str):
                    text_parts.append(sibling.strip())
            return " ".join(text_parts).strip()
        return ""

    question = get_field_by_strong("Question:")
    rationale = get_field_by_strong("Rationale for reference answer")
    candidate = get_field_by_strong("Candidate answer:")

    return question, rationale, candidate

for input_file in glob.glob(INPUT_PATTERN):
    ref_rows = []
    can_rows = []

    base_name = os.path.basename(input_file).replace("raw_", "").replace(".jsonl", "")
    output_ref_csv = f"data/ref_{base_name}.csv"
    output_can_csv = f"data/can_{base_name}.csv"

    with open(input_file, "r") as f:
        for line in f:
            ex = json.loads(line)
            html = ex.get("displayed_text", "")
            label_data = ex.get("label_annotations", {})

            question, context, candidate = extract_from_html(html)

            if not context or not question or not candidate:
                continue

            ref_rows.append({
                "question": question,
                "context": context,
                "reference": context,
                "target": "Sufficient context"
            })

            is_insufficient = (
                "insufficient_rationale" in label_data
                and any("i" in v for v in label_data["insufficient_rationale"].values())
            )

            can_rows.append({
                "question": question,
                "context": context,
                "candidate": candidate,
                "target": "Insufficient context" if is_insufficient else "Sufficient context"
            })

    pd.DataFrame(ref_rows).to_csv(output_ref_csv, index=False)
    pd.DataFrame(can_rows).to_csv(output_can_csv, index=False)

    print(f"✅ Processed {input_file}")
    print(f"   ➤ Saved {len(ref_rows)} reference examples to {output_ref_csv}")
    print(f"   ➤ Saved {len(can_rows)} candidate examples to {output_can_csv}")
