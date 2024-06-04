# Define a function to evaluate the model
def evaluate_model(dataset, model, tokenizer):
    correct_predictions = 0
    total_predictions = len(dataset)

    for example in dataset:
        input_text = example['text']
        input_question = example['question']
        actual_category = example['category']
        
        # Generate prediction
        response = train_chain.run(input_text=input_text, input_question=input_question)
        
        # Extract the predicted category from the response
        predicted_category = response.strip().split()[-1]  # Assuming the last word in the response is the category
        
        if predicted_category == actual_category:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    print(f"Model accuracy: {accuracy * 100:.2f}%")

# Evaluate the model on the tokenized dataset
evaluate_model(tokenized_dataset, llm, tokenizer)
