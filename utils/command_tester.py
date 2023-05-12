import json
import numpy as np
import os
from daisy_llm.CommandHandlers import CommandHandlers

commh = CommandHandlers(self_load=True)

def load_embeddings():
    data = {}
    path = 'utils/output/'
    for filename in os.listdir(path):
        if not filename.endswith('.json'):
            continue
        with open(path+filename, 'r') as f:
            module_data = json.load(f)
            embeddings_list = module_data['embeddings']
            command_name = module_data['module']['name']
            data[command_name] = {
                'argument': module_data['module']['argument'],
                'embeddings': [embedding['embedding'] for embedding in embeddings_list],
            }
    return data

def main():
    # Load embeddings from files
    data = load_embeddings()

    if data:

        # Output available commands
        commh.print_available_commands(data)

        while True:
            # Accept goal as input from the user
            task = input("Enter your task: ")

            # Convert goal to a list of sentence embeddings
            task_vec = commh.embed_string(task, commh.tokenizer, commh.model)

            if task_vec is None:
                print("Task contains no valid words. Please enter a different goal.")
                return False

            # Find the command with the smallest cosine distance to the goal
            (best_command, best_command_argument, best_command_confidence, next_best_command, next_best_command_argument, next_best_command_confidence) = commh.find_best_command(task_vec, data)

            # Output the best command for achieving the goal
            print("Task:", task)

            # Output the next best command for achieving the goal
            commh.print_results(
                best_command, 
                best_command_confidence, 
                next_best_command, 
                next_best_command_confidence)

if __name__ == '__main__':
    main()
