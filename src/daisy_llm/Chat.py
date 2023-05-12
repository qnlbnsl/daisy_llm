import openai
import logging
import nltk.data
import threading
import time
import yaml
import json
import requests
import re

from .ChatSpeechProcessor import ChatSpeechProcessor
from .SoundManager import SoundManager
from .Text import print_text, delete_last_lines
from .CommandHandlers import CommandHandlers
import pprint




class Chat:
	description = "Implements a chatbot using OpenAI's GPT-3 language model and allows for interaction with the user through speech or text."

	def __init__(self, ml=None, ch=None):
		self.ml = ml
		self.ch = ch
		self.csp = ChatSpeechProcessor()
		self.sounds = SoundManager()

		self.commh = CommandHandlers(self.ml)
		self.commh.load_bert_model()
		self.commh.data = self.commh.load_embeddings()

		with open("configs.yaml", "r") as f:
			self.configs = yaml.safe_load(f)
		openai.api_key = self.configs["keys"]["openai"]

		nltk.data.load('tokenizers/punkt/english.pickle')

	def request(self,
	     messages,
		 stop_event=None,
		 sound_stop_event=None,
		 tts=None,
		 tool_check=False,
		 model="gpt-3.5-turbo",
		 silent=False,
		 response_label=True,
		 temperature = 0.7
		 ):
		#Handle LLM request. Optionally convert to sentences and queue for tts, if needed.

		#Queues for handling chunks, sentences, and tts sounds
		sentences = [[]]  # create a queue to hold the sentences



		if not stop_event:
			stop_event = threading.Event()
		if not sound_stop_event:
			sound_stop_event = threading.Event()

		#Flags for handling chunks, sentences, and tts sounds
		sentence_queue_canceled = [False]  # use a list to make response_canceled mutable
		sentence_queue_complete = [False]	# use a list to make response_complete mutable

		threads = []  # keep track of all threads created
		text_stream = [""]
		return_text = [""]

		#pprint.pprint(messages)
		try:
			logging.info("Sending request to OpenAI model...")
			response = openai.ChatCompletion.create(
				model=model,
				messages=messages,
				temperature=temperature,
				stream=True,
				request_timeout=5,
			)

			#Handle chunks. Optionally convert to sentences for sentence_queue, if needed.
			arguments = {
				'response': response,
				'text_stream': text_stream,
				'sentences': sentences,
				'sentence_queue_canceled': sentence_queue_canceled,
				'sentence_queue_complete': sentence_queue_complete,
				'return_text': return_text,
				'stop_event': stop_event,
				'sound_stop_event': sound_stop_event,
				'silent': silent,
				'model': model,
				'response_label': response_label
			}
			t = threading.Thread(target=self.stream_queue_sentences, args=(arguments,))
			t.start()
			threads.append(t)

			if tts:
				self.csp.queue_and_tts_sentences(
					tts=tts, 
					sentences=sentences, 
					sentence_queue_canceled=sentence_queue_canceled, 
					sentence_queue_complete=sentence_queue_complete, 
					stop_event=stop_event, 
					sound_stop_event=sound_stop_event
					)

			while not return_text[0] and not stop_event.is_set():
				time.sleep(0.1)  # wait a bit before checking again

			# return response_complete and return_text[0] when return_text is set

			t.join()

			return return_text[0]
		
		# Handle different types of errors that may occur when sending request to OpenAI model
		except openai.error.Timeout as e:
			logging.error(f"Timeout: {e}")
			#self.csp.tts("TimeoutError Error. Check your internet connection.")
			return False  
		except openai.error.APIConnectionError as e:
			logging.error(f"APIConnectionError: {e}")
			#self.csp.tts("APIConnectionError. Sorry, I can't talk right now.")
			return False  
		except openai.error.InvalidRequestError as e:
			logging.error(f"Invalid Request Error: {e}")
			#self.csp.tts("Invalid Request Error. Sorry, I can't talk right now.")
			return False        
		except openai.APIError as e:
			logging.error(f"API Error: {e}")
			#self.csp.tts("API Error. Sorry, I can't talk right now.")
			return False
		except openai.error.RateLimitError as e:
			logging.error(f"RateLimitError: {e}")
			#self.csp.tts("Rate Limit Error. Sorry, I can't talk right now.")
			return False
		except ValueError as e:
			logging.error(f"Value Error: {e}")
			#self.csp.tts("Value Error. Sorry, I can't talk right now.")
			return False    
		except TypeError as e:
			logging.error("Type Error:"+e)
			#self.csp.tts("Type Error. Sorry, I can't talk right now.")
			return False  

	def determine_and_run_commands(
			self, 
			messages=None, 
			stop_event=None, 
			response_label=True,
			model='gpt-3.5-turbo'
			):
		output = "There was an error."
		logging.info("Checking for tool forms...")

		#HOOK: Chat_request_inner
		#Right now, only one hook can be run at a time. If a hook returns a value, the rest of the hooks are skipped.
		#I may update this soon to allow for inline responses (For example: "5+5 is [Calculator: 5+5]")
		if self.ml:
			hook_instances = self.ml.get_hook_instances()
			logging.debug(hook_instances)

			run_commands_messages = []
			if "Chat_request_inner" in hook_instances:
				last_message = messages[-1]['content']
				#goal = self.get_goal_from_conversation(messages, stop_event)
				(best_command, 
     			best_command_argument, 
			    best_command_description,
				best_command_confidence, 
				next_best_command, 
				next_best_command_argument, 
				next_best_command_confidence,
				next_best_command_description
				) = self.commh.determine_command(last_message)
				
				#Get the argument
				prompt = "1. Respond with the necessary argument to accomplish the task.\n"
				prompt += "2. If the command is incorrect to accomplish the task, respond with 'Incorrect command'.\n"
				prompt += "3. Provide only the argument as specified in Argument'.\n"
				prompt += "Task: "+last_message+"\n"
				prompt += "Command: "+best_command+"\n"
				prompt += "Description: "+best_command_description+"\n"
				prompt += "Argument format: "+best_command_argument+"\n"


				print_text(prompt)

				print_text("ARGUMENT ("+best_command+"): ", "green", "\n")

				message = [self.ch.single_message_context('system', prompt, False)]

				response = self.request(
						messages=message, 
						model="gpt-4", #Best at choosing tools
						stop_event=stop_event, 
						response_label=False
					)
				'''
				while goal is not None:

					prompt = self.build_commands_checker_prompt(goal=goal)
					temp_messages = run_commands_messages.copy()
					message = self.ch.single_message_context('user', prompt, False)
					temp_messages.append(message)

					#print(run_commands_messages)
					try:
						response = self.request(
							messages=temp_messages, 
							model="gpt-4", #Best at choosing tools
							stop_event=stop_event, 
							response_label=response_label
						)
					except Exception as e:
						logging.error("Daisy request error: "+ str(e))
						return "chatCompletion error"

					data = None
					if response:
						data = self.get_json_data(response)

					if data:
						#if data, then add it to context
						print_text("USER:", "red", "", "bold")
						print_text(prompt, None, "\n")
						run_commands_messages.append(self.ch.single_message_context('user', message, False))
						run_commands_messages.append(self.ch.single_message_context('assistant', response, False))

						command = data[0]['command']
						if command == "Goal Accomplished":
							text = "Goal Accomplished"
							print_text("USER:", "red", "", "bold")
							print_text(text, None, "\n")
							
							run_commands_messages.append(self.ch.single_message_context('user', text, False))
							break
						for module in self.ml.get_available_modules():
							if "tool_form_name" in module:
								if module["tool_form_name"] == command:

									class_name = module["class_name"]
									chat_request_inner_hook_instances = self.ml.get_hook_instances()["Chat_request_inner"]
									for instance in chat_request_inner_hook_instances:
										if instance.__class__.__name__ == class_name.split(".")[-1]:
											logging.info("Found instance: "+instance.__class__.__name__)

											#Get the argument
											prompt = "1. To achieve this goal: "+goal+"\n"
											prompt += "2. Respond to 'Answer' with the argument required for the following tool:\n"
											prompt += "	Command: "+module["tool_form_name"]+"\n"
											prompt += "	Description: "+module["tool_form_description"]+"\n"
											prompt += "	Argument: <"+module["tool_form_argument"]+">\n"
											prompt += "The argument is: "

											print_text("USER:", "red", "", "bold")
											print_text(prompt, None, "\n")
											run_commands_messages.append(self.ch.single_message_context('user', prompt, False))
											try:
												response = self.request(
													#model="gpt-4", #Much better at following the rules
													messages=run_commands_messages,
													stop_event=stop_event
													)
											except Exception as e:
												logging.error("Daisy request error: "+ str(e))
												break

											if not response:
												logging.error("Daisy request error: No response")
												break

											arg = response

											#Run the command
											print_text("Tool: ", "green")
											print_text(module["tool_form_name"] + " (" + arg + ")", None, "\n\n")

											output = "Below is the information from the command: "+module["tool_form_name"]+". Use it to accomplish the goal: "+ goal +"\n"
											output += instance.main(arg, stop_event)
											print_text("USER:", "red", "", "bold")
											print_text(output, None, "\n")
											run_commands_messages.append(self.ch.single_message_context('user', output, False))
											print("END OF TOOL")
				

			prompt = "Provide the answer based on the output above that answers the goal: "+goal
			print_text("USER:", "red", "", "bold")
			print_text(prompt, None, "\n")
			run_commands_messages.append(self.ch.single_message_context('user', prompt, False))
			print("ONE")
			try:
				result = self.request(
					model="gpt-4", #Much better at incorporating SYSTEM messages into its context
					messages=run_commands_messages,
					stop_event=stop_event
					)
			except Exception as e:
				logging.error("Daisy request error: "+ str(e))
				return "chatCompletion error"

			print("TWO")
			output = "Below is the response from the user's request. Don't mention this messages existence in your conversation. Use the inormation below in your response.\n"
			output += result


		else:
			logging.info("No data found.")
			output = "Could not cooperate with LLM to determine commands to run for goal: "+goal
		print("THREE")
		return output
		'''
	
	def get_goal_from_conversation(self, messages, stop_event):
		#Get the argument
		prompt = "Respond with the goal the user is trying to accomplish. Do not answer the question or have a conversation. Limit prose.\n"
		prompt += "\n"
		prompt += "Conversation:\n"
		#Get the last three messages and add them to the prompt
		last_three_messages = messages[-3:]
		for message in last_three_messages:
			prompt += message['role'].upper()+": "
			prompt += message['content']+"\n"
		prompt += "Goal: "

		print(prompt)
		message = self.ch.single_message_context('user', prompt, False)
		try:
			response = self.request(
				messages=[message],
				stop_event=stop_event,
				temperature = 0
				)
		except Exception as e:
			logging.error("Daisy request error: "+ str(e))
			return None

		if not response:
			logging.error("Daisy request error: No response")
			return None

		return response

	def get_json_data(self, string):
		data = None
		start_index = string.find('[')
		if start_index >= 0:
			end_index = string.rfind(']')+1
			json_data = string[start_index:end_index]
			try:

				try:
					data = json.loads(json_data)
					logging.info('Data:' + str(data))
				except json.decoder.JSONDecodeError as e:
					# Input string contains errors, attempt to fix them
					logging.error('JSONDecodeError:', e)
					
					# Search for keys with missing values
					match = json_data.search(json_data)
					if match:
						# Replace missing values with empty strings
						fixed_str = json_data[:match.end()] + '""' + json_data[match.end()+1:]
						logging.warning('Fixed input:', str(fixed_str))
						try:
							data = json.loads(fixed_str)
							logging.info('Data:'+ str(data))
						except json.decoder.JSONDecodeError:
							logging.error('Could not fix input')
					else:
						logging.error('Could not fix input')

			except json.decoder.JSONDecodeError as e:
				logging.error("JSONDecodeError: "+str(e))
				data = None
			if data and data[0] == "None":
				data = None
		else:
			logging.warning("No JSON data found in string.")
			data = None
		return data

	def build_commands_checker_prompt(self, goal=None):
						#Create a tool-chooser prompt
		prompt = "GOAL: "+goal+"\n"
		prompt += "1. Refer to previous messages in the conversation to determine what has already been accomplished.\n"
		prompt += "2. \"Commands\" contains a list of available tools for you to use.\n"
		prompt += "3. Choose the next command to be run to achieve the goal. Or choose \"Do nothing\" command if an LLM should be able to answer on its own.\n"
		prompt += "4. Format your response using JSON.\n"
		prompt += "5. Your response will be parsed in a computer program. Do not include any additional text in your response.\n"
		prompt += "6. Do not include argumants or any extra information in your JSON response.\n"
		prompt += "\n"
		prompt += "	Commands:\n"
		i=1
		for module in self.ml.get_available_modules():

			if 'tool_form_name' in module:
				prompt += f"		{i}. "
				prompt += f"Name: {module['tool_form_name']}\n"
				i+=1
				if 'tool_form_description' in module:
					prompt += f"		Description: {module['tool_form_description']}\n"
				prompt += "\n"
		i+=1
		prompt += "		"+str(i)+". Name: Goal Accomplished\n"
		prompt += "		Description: The goal has been accomplished\n"

		prompt += "You should only respond in JSON format as described below. Only one command can be chosen at a time. Do not include arguments or any value other than 'command'. \n"
		prompt += "Response Format: \n"
		prompt += "[{\n"
		prompt += '	"command": "name"\n'
		prompt += '}]\n'
		prompt += "\n"
		prompt += "Ensure the response can be parsed by Python json.loads"

		return prompt


	def stream_queue_sentences(self, arguments_dict):
		response = arguments_dict['response']
		text_stream = arguments_dict['text_stream']
		sentences = arguments_dict['sentences']
		sentence_queue_canceled = arguments_dict.get('sentence_queue_canceled', [False])
		sentence_queue_complete = arguments_dict.get('sentence_queue_complete', [False])
		return_text = arguments_dict['return_text']
		stop_event = arguments_dict['stop_event']
		sound_stop_event = arguments_dict['sound_stop_event']
		silent = arguments_dict['silent']
		model = arguments_dict['model']
		response_label = arguments_dict['response_label']

		collected_chunks = []
		collected_messages = []

		try:
			if not silent and response_label:
				print_text("Daisy ("+model+"): ", "blue", "", "bold")

			for chunk in response:
				if not sentence_queue_canceled[0]:
					if not stop_event.is_set():
						temp_sentences = []
						collected_chunks.append(chunk)
						chunk_message = chunk['choices'][0]['delta']
						collected_messages.append(chunk_message)
						text_stream[0] = ''.join([m.get('content', '') for m in collected_messages])
						logging.debug(text_stream[0])

						if not silent:
							if 'content' in chunk_message:
								print_text(chunk_message['content'])
						
						#Tokenize the text into sentences
						temp_sentences = self.csp.nltk_sentence_tokenize(text_stream[0])
						sentences[0] = temp_sentences  # put the sentences into the queue
					else:
						sentence_queue_canceled[0] = True
						logging.info("Sentence queue canceled")
						return
			print_text("\n\n")
		except requests.exceptions.ConnectionError as e:
			logging.error("stream_queue_sentences(): Request timeout. Check your internet connection.")
			sentence_queue_canceled[0] = True

		time.sleep(0.01)
		sentence_queue_complete[0] = True
		return_text[0] = text_stream[0]
		sound_stop_event.set()
		logging.info("Sentence queue complete")
		return


	def display_messages(self, chat_handlers):
		"""Displays the messages stored in the messages attribute of ContectHandlers."""
		for message in chat_handlers.get_context():
			# Check if the message role is in the list of roles to display
			print(f"{message['role'].upper()}: {message['content']}\n\n")