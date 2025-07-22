from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

MODEL_NAME = "trained_model"

label_set = {"Sufficient", "Rationale insufficient", "Question incomplete"}

def clean_prediction(pred):
    pred = pred.strip().lower()
    for label in label_set:
        if pred == label.lower():
            return label
    return "Sufficient"

def load_and_prepare_data():
    full_df = pd.read_csv("data/processed.csv")
    only_q_df = pd.read_csv("data/processed_q_only.csv")

    full_df["input"] = (
        "question: " + full_df["question"] +
        " reference answer: " + full_df["reference"] +
        " rationale: " + full_df["rationale"] +
        " candidate answer: " + full_df["candidate"]
    )
    full_df["target"] = full_df["label"]

    only_q_df["input"] = "question: " + only_q_df["question"]
    only_q_df["target"] = only_q_df["label"]

    combined_df = pd.concat([full_df, only_q_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=["target"])

    minority_df = combined_df[combined_df["target"] != "Sufficient"]
    sufficient_df = combined_df[combined_df["target"] == "Sufficient"]

    downsampled_sufficient = sufficient_df.sample(n=len(minority_df), random_state=42)
    balanced_df = pd.concat([minority_df, downsampled_sufficient])
    return balanced_df.reset_index(drop=True)

def preprocess(batch, tokenizer):
    model_inputs = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(batch["target"], truncation=True, padding="max_length", max_length=64)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def evaluate():
    df = load_and_prepare_data()
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    eval_loader = DataLoader(tokenized["test"], batch_size=4, collate_fn=data_collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_eval_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=16)

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

            cleaned_preds = [clean_prediction(p) for p in preds]
            cleaned_labels = [clean_prediction(t) for t in targets]

            all_preds.extend(cleaned_preds)
            all_labels.extend(cleaned_labels)

    avg_eval_loss = total_eval_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    print(f"Exact Match Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, digits=3))

    for i in range(5):
        print(f"\n🔍 Example {i+1}")
        print(f"Prediction: {all_preds[i]}")
        print(f"Target    : {all_labels[i]}")

if __name__ == "__main__":
    evaluate()
