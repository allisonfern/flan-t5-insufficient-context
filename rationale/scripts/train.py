import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5ForSequenceClassification, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import ast


MODEL_NAME = "google/flan-t5-small"

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



def train():
    df = load_and_prepare_data()
    train_df, _ = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    tokenized = train_dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(tokenized, batch_size=16, shuffle=True, collate_fn=data_collator)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification"
    )
    model.to(device)
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    model.save_pretrained("../trained_model_a")
    tokenizer.save_pretrained("../trained_model_a")


if __name__ == "__main__":
    train()