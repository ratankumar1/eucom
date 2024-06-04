from langchain_community.llms.vllm import VLLM
from langchain.chains import LLMChain
from langchain.prompts import SimplePromptTemplate

# Define your VLLM model and API key
api_key = "your_vllm_api_key"
model_name = "your_model_name"

# Initialize the VLLM model
llm = VLLM(api_key=api_key, model=model_name)

# Define a simple prompt template for training
prompt_template = SimplePromptTemplate(
    input_variables=["text", "question"],
    template=(
        "Text: {text}\n"
        "Question: {question}\n"
        "Category:"
    )
)

# Create a chain using the VLLM model and the prompt template
train_chain = LLMChain(llm=llm, prompt_template=prompt_template)

# Fine-tuning the model using the dataset
for example in tokenized_dataset:
    input_text = example['text']
    input_question = example['question']
    target_category = example['category']
    train_chain.run(input_text=input_text, input_question=input_question, target_category=target_category)

# Save the model after training
llm.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
