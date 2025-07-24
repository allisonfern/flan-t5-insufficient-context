from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import T5Tokenizer, T5ForSequenceClassification, DataCollatorWithPadding

MODEL_NAME = "trained_model"

LABEL2ID = {
    "Sufficient": 0,
    "Rationale insufficient": 1,
    "Question incomplete": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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
    tokenized = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = [LABEL2ID[label] for label in batch["target"]]
    return tokenized


def evaluate():
    df = load_and_prepare_data()
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForSequenceClassification.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
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
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            total_eval_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_eval_loss = total_eval_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=[ID2LABEL[i] for i in range(3)], digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[ID2LABEL[i] for i in range(3)],
                yticklabels=[ID2LABEL[i] for i in range(3)])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    # Show a few predictions
    for i in range(10):
        print(f"\n🔍 Example {i+1}")
        print(f"Prediction: {ID2LABEL[all_preds[i]]}")
        print(f"Target    : {ID2LABEL[all_labels[i]]}")


if __name__ == "__main__":
    evaluate()
