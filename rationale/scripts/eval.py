from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import T5Tokenizer, T5ForSequenceClassification, DataCollatorWithPadding

MODEL_NAME = "../trained_model_a"

LABEL2ID = {
    "Sufficient": 0,
    "Rationale insufficient": 1,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_and_prepare_data():
    mmar_df = pd.read_csv("../data/processed_mmar_a.csv")
    mmau_df = pd.read_csv("../data/processed_mmaumini_a.csv")

    full_df = pd.concat([mmar_df, mmau_df], ignore_index=True)

    full_df["input"] = full_df["question"]
    full_df["target"] = full_df["label"]


    sufficient_df = full_df[full_df["target"] == "Sufficient"]
    insufficient_df = full_df[full_df["target"] == "Rationale insufficient"]
    print("Sufficient:", len(sufficient_df))
    print("Insufficient due to rationale:", len(insufficient_df))

    max_size = max(len(sufficient_df), len(insufficient_df))
    # Upsample minority class
    sufficient_df_upsampled = resample(sufficient_df, replace=True, n_samples=max_size, random_state=42)
    insufficient_df_upsampled = resample(insufficient_df, replace=True, n_samples=max_size, random_state=42)

    balanced_df = pd.concat([
        sufficient_df_upsampled,
        insufficient_df_upsampled,
    ])
    
    return balanced_df.reset_index(drop=True)


def preprocess(batch, tokenizer):
    tokenized = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = [LABEL2ID[label] for label in batch["target"]]
    return tokenized

def build_test_loader(model_path=MODEL_NAME, batch_size=4):
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    df = load_and_prepare_data()
    _, test_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)

    test_df["input"] = ("question: " + test_df["question"])

    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test = test_dataset.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=test_dataset.column_names)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_loader = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return eval_loader, tokenizer, test_df["input"].tolist()

def evaluate():
    eval_loader, tokenizer, inputs = build_test_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification"
    )
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
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            total_eval_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_eval_loss = total_eval_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"\nEvaluation Loss: {avg_eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=[ID2LABEL[i] for i in range(2)], digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[ID2LABEL[i] for i in range(2)],
                yticklabels=[ID2LABEL[i] for i in range(2)])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("../confusion_matrix_a.png")

    # Show a few misclassified examples
    for i in range(10):
        print(f"\n🔍 Example {i+1}")
        print(f"Input: {inputs[i]}")
        print(f"Prediction: {ID2LABEL[all_preds[i]]}")
        print(f"Target    : {ID2LABEL[all_labels[i]]}")


if __name__ == "__main__":
    evaluate()
