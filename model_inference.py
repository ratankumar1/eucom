from transformers import pipeline

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./trained_model")
tokenizer = AutoTokenizer.from_pretrained("./trained_model")

# Create a pipeline for text classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Perform inference
text = "Shakespeare wrote many famous plays, including Romeo and Juliet."
question = "Who wrote Romeo and Juliet?"
input_text = tokenizer(text, question, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

# Get prediction
prediction = classifier(input_text)
print(f"Predicted category: {prediction}")
