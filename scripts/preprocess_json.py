import json
import re
import pandas as pd
from bs4 import BeautifulSoup

INPUT_JSON = "data/raw_mmau_pilot_af3.jsonl"
OUTPUT_CSV = "data/mmau_pilot_af3.csv"

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

rows = []

with open(INPUT_JSON, "r") as f:
    for line in f:
        ex = json.loads(line)
        html = ex.get("displayed_text", "")
        label_data = ex.get("label_annotations", {})

        question, context, candidate = extract_from_html(html)

        # Only include examples with valid context
        if not context or not question or not candidate:
            continue

        is_insufficient = (
            "insufficient_rationale" in label_data
            and any("i" in v for v in label_data["insufficient_rationale"].values())
        )

        input_text = f"question: {question} context: {context} candidate answer: {candidate}"
        target_text = "Insufficient context" if is_insufficient else "Sufficient context"

        rows.append({
            "input": input_text,
            "target": target_text
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(df.columns)
print(f"✅ Saved {len(df)} examples to {OUTPUT_CSV}")