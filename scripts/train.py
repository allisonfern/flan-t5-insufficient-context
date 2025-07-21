import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

MODEL_NAME = "google/flan-t5-small"

df = pd.read_csv("data/processed.csv")

print(df.columns)

# candidate answers: human annotation for sufficiency (already saved in target)
df["input"] = (
    "question: " + df["question"] +
    " reference answer: " + df["reference"] +
    " rationale: " + df["rationale"] +
    " candidate answer: " + df["candidate"]

)

df["target"] = df["label"]

df = df[["input", "target"]].dropna()
df.reset_index(drop=True, inplace=True)  # drop old index

print(df.head())
print(df.columns)



# Training
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1) # 90% training 10% testing ?

# load flan t5 huggingface tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def preprocess(batch):
    model_inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    labels = tokenizer(
        batch["target"],
        truncation=True,
        padding="max_length",
        max_length=64
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# custom training loop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

train_loader = DataLoader(tokenized["train"], batch_size=4, shuffle=True, collate_fn=data_collator)
eval_loader = DataLoader(tokenized["test"], batch_size=4, collate_fn=data_collator) # maybe increase batch size when i get more data

optimizer = AdamW(model.parameters(), lr=5e-5) # same optimizer as Trainer class

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        # using t4 gpu for now
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    # Evaluation loop
    model.eval()
    total_eval_loss = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Epoch {epoch+1} - Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(eval_loader)
    print(f"Epoch {epoch+1} Evaluation Loss: {avg_eval_loss:.4f}")


