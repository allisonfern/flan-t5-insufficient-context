import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

MODEL_NAME = "google/flan-t5-small"

# combines human annotation data with question only data, downsamples sufficient label, returns balanced_df
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

def train():
    df = load_and_prepare_data()
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_loader = DataLoader(tokenized["train"], batch_size=4, shuffle=True, collate_fn=data_collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
