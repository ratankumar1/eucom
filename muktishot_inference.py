from langchain_community.llms.vllm import VLLM
from langchain.chains import ConversationChain
from langchain.prompts import SimplePromptTemplate

# Define your VLLM model and API key
api_key = "your_vllm_api_key"
model_name = "your_model_name"

# Initialize the VLLM model
llm = VLLM(api_key=api_key, model=model_name)

# Define a multi-shot prompt template with examples for each category
prompt_template = SimplePromptTemplate(
    input_variables=["text", "question"],
    template=(
        "Text: Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.\n"
        "Question: What is Alice doing?\n"
        "Category: Characters\n\n"
        
        "Text: The quick brown fox jumps over the lazy dog.\n"
        "Question: What is the color of the fox?\n"
        "Category: Animals\n\n"
        
        "Text: The Eiffel Tower is located in Paris, France.\n"
        "Question: Where is the Eiffel Tower located?\n"
        "Category: Locations\n\n"
        
        "Text: Hydrogen is the first element on the periodic table.\n"
        "Question: What is the first element on the periodic table?\n"
        "Category: Science\n\n"
        
        "Text: The capital of Japan is Tokyo.\n"
        "Question: What is the capital of Japan?\n"
        "Category: Geography\n\n"
        
        "Text: The Battle of Hastings was fought in 1066.\n"
        "Question: When was the Battle of Hastings fought?\n"
        "Category: History\n\n"
        
        "Text: The novel '1984' was written by George Orwell.\n"
        "Question: Who wrote the novel '1984'?\n"
        "Category: Literature\n\n"
        
        "Text: Beethoven composed the Fifth Symphony.\n"
        "Question: Who composed the Fifth Symphony?\n"
        "Category: Music\n\n"
        
        "Text: Apple is known for its iPhones and Mac computers.\n"
        "Question: What is Apple known for?\n"
        "Category: Technology\n\n"
        
        "Text: {text}\n"
        "Question: {question}\n"
        "Category:"
    )
)

# Create a conversation chain using the VLLM model and the prompt template
conversation = ConversationChain(llm=llm, prompt_template=prompt_template)

# Define your text and question
text = "Shakespeare wrote many famous plays, including Romeo and Juliet."
question = "Who wrote Romeo and Juliet?"

# Generate a response
response = conversation.run(input={"text": text, "question": question})

# Print the response
print(response)
