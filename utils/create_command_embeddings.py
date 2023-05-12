import json
import os.path
from transformers import AutoTokenizer, AutoModel
from daisy_llm.CommandHandlers import CommandHandlers

#INSTRUCTIONS
#1. Run this script from the command line
#2. Enter the name of the tool you want to add embeddings for
#3. Enter example search terms (paste multiple lines, leave a blank line to finish)
    #Effective prompt: Provide examples that cover a wide range of variations and potential queries related to the following task: Get the weather forecast
#4. Enter the module description
#5. Enter the module argument
#6. Repeat steps 2-5 for each tool you want to add embeddings for


commh = CommandHandlers(True)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#json_file = 'embeddings.json'

filename_prefix = 'module-'
path = 'utils/output/'


def load_embeddings(embeddings_file):
    # Check if embeddings file exists, create it if not
    if not os.path.isfile(embeddings_file):
        with open(embeddings_file, 'w') as f:
            json.dump({}, f)

    # Load embeddings from JSON file
    with open(embeddings_file, 'r') as f:
        return json.load(f)


def add_tool(embeddings, tool_name, testing=False):
    os.makedirs(path, exist_ok=True)
    file_name = f"{path}{filename_prefix}{tool_name}.json"

    module_name = None
    module_description = None
    module_argument = None
    examples = []

    if os.path.isfile(file_name):
        # Load existing embeddings from file
        with open(file_name, 'r') as f:
            data = json.load(f)
            module_name = data['module']['name']
            module_description = data['module']['description']
            module_argument = data['module']['argument']
            examples = data['embeddings']

    print("Enter example search terms (paste multiple lines, leave a blank line to finish):")
    while True:
        line = input()
        if not line:
            break

        lines = line.strip().split('\n')
        for example_line in lines:
            example = example_line.strip()
            example_embedding = commh.embed_string(example, tokenizer, model)
            if example_embedding is not None:
                examples.append({'text': example, 'embedding': example_embedding.tolist()})
            else:
                print("Skipping invalid embedding for example:", example)

            if testing:
                break

    if examples:
        if not module_description:
            module_description = input("Enter the module description: ")

        if not module_argument:
            module_argument = input("Enter the module argument: ")

        embeddings[tool_name] = {
            'module': {
                'name': tool_name,
                'description': module_description,
                'argument': module_argument
            },
            'embeddings': examples
        }
        print(f"{len(examples)} examples added to {tool_name}.")
    else:
        print(f"No valid example embeddings found for {tool_name}. Skipping tool.")

    return embeddings



def save_embeddings(embeddings, tool_name, output_dir=path):
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}{filename_prefix}{tool_name}.json"

    # Merge existing embeddings with new embeddings
    new_embeddings = embeddings[tool_name]['embeddings']
    
    # Convert the embeddings to lists before outputting
    embeddings_copy = new_embeddings.copy()
    for example in embeddings_copy:
        example_embedding = example['embedding']
        example['embedding'] = example_embedding

    
    # Save combined embeddings to file
    with open(file_name, 'w') as f:
        json.dump({
            'module': embeddings[tool_name]['module'],
            'embeddings': embeddings_copy
        }, f, indent=4)
    print(f"Embeddings saved to {file_name} file.")


def run_prompt():
    embeddings = {}
    while True:
        command = input("Enter a command (add, list, save, quit): ")
        if command == 'add':
            tool_name = input("Enter a tool name: ")
            add_tool(embeddings, tool_name)
        elif command == 'list':
            commh.list_tools(embeddings)
        elif command == 'save':
            for tool_name in embeddings:
                save_embeddings(embeddings, tool_name)
            break
        elif command == 'quit':
            save_response = input("Do you want to save your changes before exiting? (y/n) ")
            if save_response.lower() == 'y':
                for tool_name in embeddings:
                    save_embeddings(embeddings, tool_name)
            break
        else:
            print("Invalid command. Please try again.")


if __name__ == "__main__":
    run_prompt()
