import logging
import json
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

class CommandHandlers:

    def __init__(self, ml=None):
        self.ml = ml
        self.enabled_modules = []
        self.tokenizer, self.model = self.load_bert_model('bert-base-uncased')


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
                        'embeddings': [embedding['embedding'] for embedding in embeddings_list],
                    }
        return data


    def load_bert_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model


    def embed_string(self, string, tokenizer, model):
        input_ids = torch.tensor([tokenizer.encode(string)])
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Shape: [batch_size, sequence_length, hidden_size]
            embeddings = torch.mean(last_hidden_states, dim=1)  # Take the mean of the sequence to get a single vector
        return embeddings[0]


    def compute_distance(self, goal_vec, command_mean_vec):
        distance = cosine(goal_vec, command_mean_vec)
        return distance


    def find_best_command(self, goal_vec, embeddings):
        best_command = None
        best_command_distance = float("inf")
        best_command_confidence = 0.0
        best_command_argument = None
        next_best_command = None
        next_best_command_distance = float("inf")
        next_best_command_confidence = 0.0
        next_best_command_argument = None

        for command_name in embeddings.keys():
            command_distance = 0.0
            command_data = embeddings[command_name]['embeddings']
            for emb in command_data:
                command_distance += cosine(goal_vec, emb)
            command_distance /= len(command_data)
            print(command_name, command_distance)
            command_confidence = (1 - command_distance) * 100
            argument = embeddings[command_name]['argument']


            if command_confidence > best_command_confidence:
                next_best_command = best_command
                next_best_command_confidence = best_command_confidence
                next_best_command_argument = best_command_argument
                best_command_confidence = command_confidence
                best_command = command_name
                best_command_argument = argument
            elif command_confidence < next_best_command_confidence:
                next_best_command_confidence = command_confidence
                next_best_command = command_name
                next_best_command_argument = argument

        return best_command, best_command_argument, best_command_confidence, next_best_command, next_best_command_argument, next_best_command_confidence


    def print_available_commands(self, embeddings):
        print("Available commands:")
        for command in embeddings.keys():
            print(f"- {command}")

    def determine_command(self, task, threshold=0.5):
        # Load embeddings from files
        data = self.load_embeddings()

        if data:
            # Output available commands
            self.print_available_commands(data)

            # Convert goal to a sentence embedding
            task_vec = self.embed_string(task, self.tokenizer, self.model)

            if task_vec is None:
                print("Task contains no valid words. Please enter a different goal.")
                return False

            # Find the command with the smallest cosine distance to the goal
            (best_command, best_command_argument, best_command_confidence, next_best_command, next_best_command_argument, next_best_command_confidence) = self.find_best_command(task_vec, data)

            # Output the best command for achieving the goal
            print("Task:", task)

            if not np.isnan(best_command_confidence):
                print(f"Best command ({best_command_confidence}%): {best_command}")

            # Output the next best command for achieving the goal
            if not np.isnan(next_best_command_confidence):
                print(f"Next best command ({next_best_command_confidence}%): {next_best_command}")

            if best_command_confidence >= threshold:
                return best_command, best_command_argument, best_command_confidence, next_best_command, next_best_command_argument, next_best_command_confidence
            else:
                return False
        else:
            return None
