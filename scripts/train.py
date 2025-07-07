import pandas as pd
from datasets import Dataset #
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

MODEL_NAME = "google/flan-t5-small"

df = pd.read_csv("data/mmau_pilot_af3.csv")

dataset = Dataset.from_pandas(df[["input", "target"]])
dataset = dataset.train_test_split(test_size=0.1)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def preprocess(example):
    model_inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(example["target"], truncation=True, padding="max_length", max_length=64)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./flan-t5-small-insufficient-context",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
)

trainer.train()
