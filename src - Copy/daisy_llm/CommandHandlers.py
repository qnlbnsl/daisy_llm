import logging
import json
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import traceback

class CommandHandlers:

    def __init__(self, ml=None, self_load=False):
        self.ml = ml
        self.enabled_modules = []
        self.tokenizer = None
        self.model = None

        if self_load:
            self.load_bert_model()



    def load_embeddings(self):
        self.enabled_modules = self.ml.get_enabled_modules()
        data = {}
        for enabled_module in self.enabled_modules:
            module_path = os.path.dirname(enabled_module.replace('.', '/'))
            embeddings_path = os.path.join(module_path, 'module.json')
            # Restructure the data for processing
            if os.path.isfile(embeddings_path):
                with open(embeddings_path, 'r') as f:
                    module_data = json.load(f)
                    embeddings_list = module_data['embeddings']
                    command_name = module_data['module']['name']
                    data[command_name] = {
                        'argument': module_data['module']['argument'],
                        'description': module_data['module']['description'],
                        'embeddings': [embedding['embedding'] for embedding in embeddings_list],
                    }
        return data


    def load_bert_model(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def list_tools(self, embeddings):
        for tool in embeddings.values():
            print(tool['module']['name'])
            for example in tool['embeddings']:
                print(f"  {example['text']}")

    def get_command_info_text(self, data):
        tool_info = ""
        for command, info in data.items():
            tool_info += f"Command: {command}\n"
            tool_info += f"Description: {info['description']}\n"
            tool_info += f"Argument format: {info['argument']} (String)\n"
            tool_info += "\n"

        return tool_info

    def embed_string(self, string, tokenizer, model):
        print("Embedding: ", string)
        input_ids = torch.tensor([tokenizer.encode(string)])
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Shape: [batch_size, sequence_length, hidden_size]
            embeddings = torch.mean(last_hidden_states, dim=1)  # Take the mean of the sequence to get a single vector
            print("Embedded")
        return embeddings[0]


    def compute_distance(self, goal_vec, command_mean_vec):
        distance = cosine(goal_vec, command_mean_vec)
        return distance


    def find_best_command(self, goal_vec, embeddings):
        best_command = None
        best_command_confidence = 0.0
        best_command_argument = None
        best_command_description = None
        next_best_command = None
        next_best_command_confidence = 0.0
        next_best_command_argument = None
        next_best_command_description = None

        for command_name in embeddings.keys():
            command_data = embeddings[command_name]['embeddings']
            command_match = 0.0
            for emb in command_data:
                match = 1 - cosine(goal_vec, emb)
                if match > command_match:
                    command_match = match
            command_confidence = command_match * 100

            if command_confidence > best_command_confidence:
                next_best_command = best_command
                next_best_command_confidence = best_command_confidence
                next_best_command_argument = best_command_argument
                next_best_command_description = best_command_description
                best_command_confidence = command_confidence
                best_command = command_name
                best_command_argument = embeddings[command_name]['argument']
                best_command_description = embeddings[command_name]['description']
            elif command_confidence > next_best_command_confidence:
                next_best_command_confidence = command_confidence
                next_best_command = command_name
                next_best_command_argument = embeddings[command_name]['argument']
                next_best_command_description = embeddings[command_name]['description']

        return (
            best_command,
            best_command_argument,
            best_command_description,
            best_command_confidence,
            next_best_command,
            next_best_command_argument,
            next_best_command_description,
            next_best_command_confidence,
        )




    def print_available_commands(self, embeddings):
        print("Available commands:")
        for command in embeddings.keys():
            print(f"- {command}")

    def print_results(self, best_command, best_command_confidence, next_best_command, next_best_command_confidence):
        if not np.isnan(best_command_confidence):
            print(f"Best command ({best_command_confidence:.2f}%): {best_command}")

        # Output the next best command for achieving the goal
        if not np.isnan(next_best_command_confidence):
            print(f"Next best command ({next_best_command_confidence:.2f}%): {next_best_command}")

    def determine_command(self, task, threshold=0.5):

        if self.data:
            # Output available commands
            self.print_available_commands(self.data)

            # Convert goal to a sentence embedding
            task_vec = self.embed_string(task, self.tokenizer, self.model)

            if task_vec is None:
                print("Task contains no valid words. Please enter a different goal.")
                return False

            # Find the command with the smallest cosine distance to the goal
            (best_command, 
             best_command_argument, 
             best_command_description,
             best_command_confidence, 
             next_best_command, 
             next_best_command_argument, 
             next_best_command_description,
             next_best_command_confidence
             ) = self.find_best_command(task_vec, self.data)

            # Output the best command for achieving the goal
            print("Task:", task)

            # Output the next best command for achieving the goal
            self.print_results(
                best_command, 
                best_command_confidence, 
                next_best_command, 
                next_best_command_confidence)
            
            return best_command, best_command_argument, best_command_description, best_command_confidence, next_best_command, next_best_command_argument, next_best_command_description, next_best_command_confidence
            
        else:
            return None
