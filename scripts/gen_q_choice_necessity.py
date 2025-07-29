import torch
from transformers import T5Tokenizer, T5ForSequenceClassification
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from typing import List
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

LABEL2ID = {
    "Sufficient": 0,
    "Rationale insufficient": 1,
    "Question incomplete": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_NAME = "trained_model_0.9113"

def preprocess_eval(example, tokenizer):
    input_text = example["question"]
    return tokenizer(input_text, truncation=True, padding="max_length", max_length=512)

def get_ground_truth_labels(df: pd.DataFrame) -> List[str]:
    return [
        "Sufficient" if val == "no" else "Question incomplete"
        for val in df["needs_choices_to_answer"]
    ]



def generate_choice_necessity() -> List[str]:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForSequenceClassification.from_pretrained(MODEL_NAME)

    df = pd.read_json("data/speech_sliced.json")
    df = df[df["question"].notna()].reset_index(drop=True)

    dataset = Dataset.from_pandas(df)
    tokenized = dataset.map(lambda x: preprocess_eval(x, tokenizer), batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(tokenized, batch_size=4, collate_fn=data_collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend([ID2LABEL[p.item()] for p in preds])

    return predictions, df

def evaluate_model(preds: List[str], gold: List[str]):
    print("=== Accuracy ===")
    print(f"{accuracy_score(gold, preds):.4f}\n")

    print("=== Classification Report ===")
    print(classification_report(gold, preds, labels=list(LABEL2ID.keys()), zero_division=0))

    # print("=== Confusion Matrix ===")
    # cm = confusion_matrix(gold, preds, labels=list(LABEL2ID.keys()))
    # print(pd.DataFrame(cm, index=[f"True: {l}" for l in LABEL2ID], columns=[f"Pred: {l}" for l in LABEL2ID]))

if __name__ == "__main__":
    predictions, filtered_df = generate_choice_necessity()
    gold_labels = get_ground_truth_labels(filtered_df)
    evaluate_model(predictions, gold_labels)

