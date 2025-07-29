import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5ForSequenceClassification, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import ast

from evaluation import evaluate



MODEL_NAME = "google/flan-t5-small"

LABEL2ID = {
    "Sufficient": 0,
    "Rationale insufficient": 1,
    "Question incomplete": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_and_prepare_data():
    mmar_df = pd.read_csv("data/reformatted/reform_processed_mmar.csv")
    mmau_df = pd.read_csv("data/reformatted/reform_processed_mmaumini.csv")
    full_df = pd.concat([mmar_df, mmau_df], ignore_index=True)
    only_q_df = pd.read_csv("data/processed_q_only.csv")

    full_df["target"] = full_df["label"]
    only_q_df["target"] = only_q_df["label"]

    combined_df = pd.concat([full_df, only_q_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=["target"]) # remove rows with no label (shouldn't be any)

    sufficient_df = combined_df[combined_df["target"] == "Sufficient"]
    insufficient_q_df = combined_df[combined_df["target"] == "Question incomplete"]
    insufficient_r_df = combined_df[combined_df["target"] == "Rationale insufficient"]
    print("Sufficient:", len(sufficient_df))
    print("Insufficient due to question:", len(insufficient_q_df))
    print("Insufficient due to rationale:", len(insufficient_r_df))

    max_size = max(len(sufficient_df), len(insufficient_q_df), len(insufficient_r_df))
    # Upsample each minority class
    sufficient_df_upsampled = resample(sufficient_df, replace=True, n_samples=max_size, random_state=42)
    insufficient_q_df_upsampled = resample(insufficient_q_df, replace=True, n_samples=max_size, random_state=42)
    insufficient_r_df_upsampled = resample(insufficient_r_df, replace=True, n_samples=max_size, random_state=42)

    balanced_df = pd.concat([
        sufficient_df_upsampled,
        insufficient_q_df_upsampled,
        insufficient_r_df_upsampled
    ])
    
    return balanced_df.reset_index(drop=True)


def dup_training_data(df):
    train_df = df.copy()

    # drop existing 'candidate' column to avoid name clash
    if "candidate" in train_df.columns:
        train_df = train_df.drop(columns=["candidate"])

    train_df_exploded = train_df.explode('candidates').reset_index(drop=True)
    train_df_exploded = train_df_exploded.rename(columns={'candidates': 'candidate'})

    train_df_exploded["input"] = (
        "question: " + train_df_exploded["question"].fillna('') +
        " reference answer: " + train_df_exploded["reference"].fillna('') +
        " rationale: " + train_df_exploded["rationale"].fillna('') +
        " candidate answer: " + train_df_exploded["candidate"].fillna('')
    )

    return train_df_exploded

def preprocess(batch, tokenizer):
    tokenized = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = [LABEL2ID[label] for label in batch["target"]]
    return tokenized




def train():
    df = load_and_prepare_data()
    train_df, _ = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)
    exploded_train_df = dup_training_data(train_df)
    train_dataset = Dataset.from_pandas(exploded_train_df)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    tokenized = train_dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(tokenized, batch_size=4, shuffle=True, collate_fn=data_collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
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

    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")


if __name__ == "__main__":
    train()