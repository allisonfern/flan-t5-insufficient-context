from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./flan-t5-small-insufficient-context")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model.eval()

# Load and format the test dataset
df = pd.read_csv("data/mmau_pilot_af3.csv")
df["input"] = "question: " + df["question"] + " context: " + df["context"]
df["target"] = df.apply(lambda row: row["answer"] if row["is_sufficient"] else "Insufficient context", axis=1)

# Just use test portion
test_df = df.sample(frac=0.1, random_state=42)  # same split as train_test_split if not saved
test_dataset = Dataset.from_pandas(test_df[["input", "target"]])

predictions = []
for item in tqdm(test_dataset):
    input_ids = tokenizer(item["input"], return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=64)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predictions.append(pred)

from sklearn.metrics import classification_report, accuracy_score

# True labels
true = ["Insufficient context" if t == "Insufficient context" else "Sufficient" for t in test_dataset["target"]]
pred = ["Insufficient context" if p == "Insufficient context" else "Sufficient" for p in predictions]

print(classification_report(true, pred))
print("Accuracy:", accuracy_score(true, pred))
