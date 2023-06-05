import openai
import logging
import nltk.data
import threading
import time
import yaml
import json
import requests
import re
import dirtyjson
import traceback

from .ChatSpeechProcessor import ChatSpeechProcessor
from .SoundManager import SoundManager
from .Text import print_text, delete_last_lines
#from .CommandHandlers import CommandHandlers
import pprint




class Chat:
	description = "Implements a chatbot using OpenAI's GPT-3 language model and allows for interaction with the user through speech or text."

	def __init__(self, ml=None):
		self.ml = ml
		self.ch = ml.ch
		self.commh = ml.commh
		self.csp = ChatSpeechProcessor()
		self.sounds = SoundManager()

		self.commh.data = self.commh.load_commands()

		with open("configs.yaml", "r") as f:
			self.configs = yaml.safe_load(f)
		openai.api_key = self.configs["keys"]["openai"]
		self.speak_thoughts = self.configs["chaining"]["speak_thoughts"]

		#nltk.data.load('tokenizers/punkt/english.pickle')

	def request(self,
	     messages,
		 stop_event=None,
		 sound_stop_event=None,
		 tts=None,
		 tool_check=False,
		 model="gpt-3.5-turbo",
		 silent=False,
		 response_label=True,
		 temperature = 0.7,
		 max_tokens = None
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

		i = 0
		while True:
			try:
				logging.info("Sending request to OpenAI model...")
				if max_tokens: #What the hell is max_tokens default value?
					response = openai.ChatCompletion.create(
						model=model,
						messages=messages,
						temperature=temperature,
						stream=True,
						request_timeout=5,
						max_tokens=max_tokens
					)
				else:
					response = openai.ChatCompletion.create(
						model=model,
						messages=messages,
						temperature=temperature,
						stream=True,
						request_timeout=5
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
			except openai.error.APIConnectionError as e:
				logging.error(f"APIConnectionError: {e}")
				#self.csp.tts("APIConnectionError. Sorry, I can't talk right now.")
			except openai.error.InvalidRequestError as e:
				logging.error(f"Invalid Request Error: {e}")
				#self.csp.tts("Invalid Request Error. Sorry, I can't talk right now.")
				return False        
			except openai.APIError as e:
				logging.error(f"API Error: {e}")
				#self.csp.tts("API Error. Sorry, I can't talk right now.")
			except openai.error.RateLimitError as e:
				logging.error(f"RateLimitError: {e}")
				#self.csp.tts("Rate Limit Error. Sorry, I can't talk right now.")
			except ValueError as e:
				logging.error(f"Value Error: {e}")
				#self.csp.tts("Value Error. Sorry, I can't talk right now.")
				return False    
			except TypeError as e:
				logging.error("Type Error:"+e)
				#self.csp.tts("Type Error. Sorry, I can't talk right now.")
				return False  
			
			i += 1
			if i == 3:
				logging.error("OpenAI model request failed 3 times. Aborting.")
				return False	

	def determine_and_run_commands(
			self, 
			messages=None, 
			stop_event=None, 
			sound_stop_event=None,
			model='gpt-3.5-turbo',
			sensitivity=0.5,
			tts=None
			):
		logging.info("Checking for tool forms...")

		if not stop_event:
			stop_event = threading.Event()

		#Get the task, if any
		print_text("Task: ", "yellow")
		task = self.get_task_from_conversation(messages, stop_event)
		if not task:
			return
		
		else:

			if self.ml:
				hook_instances = self.ml.get_hook_instances()
				logging.debug(hook_instances)

				if "Chat_request_inner" in hook_instances:

					command = None
					command_argument = None
					#command_argument = self.get_command_argument(task, command, description, argument_format, stop_event)

					task_complete_answer = None
					ask_question = None
					reasoning_context = []
					while True:
						if command:
							#if str(command_argument).lower() != "incorrect command":
							#Run the command, get the output
							for instance in hook_instances["Chat_request_inner"]:
								class_name = type(instance).__name__
								if command == class_name:
									module_output = "[Output from "+class_name+": "+command_argument+"]\n"
									module_output += instance.main(command_argument, stop_event)+"\n\n"
									module_output = module_output #Limit output size so we don't immediately run out of context space
									reasoning_context.append(self.ch.single_message_context("user", module_output, False))
									print_text("Chaining (Module Output): ", "yellow")
									print_text(module_output, None, "\n")

									command = None
									break
						
						#Check for task completion
						task_complete_answer = self.check_for_task_completion(task, reasoning_context, stop_event)
						if task_complete_answer:
							break
						
						#Check validity and determine next steps
						validity_prompt = self.validity_prompt(task, stop_event)
						reasoning_context_copy = reasoning_context.copy()
						reasoning_context_copy.append(self.ch.single_message_context("user", validity_prompt, False))
						#print(reasoning_context_copy)
						print_text("Chaining (Assistant Reasoning): ", "yellow")
						response = self.request(
							messages=reasoning_context_copy, 
							model="gpt-4", #Best at choosing tools
							stop_event=stop_event, 
							response_label=False
						)

						#Truncate long module output (USER) to save on context space		
						for item in reasoning_context:
							if item['role'] == "user":
								content = item['content']
								if len(content) > 1000:
									content = content[:975] + "...[Message truncated]"
									item['content'] = content

						#Get the subtask for an incomplete task
						try:
							data = dirtyjson.loads(response)
							command = data['thoughts']['command']
							command_argument = str(data['thoughts']['argument'])
							thought = data['thoughts']['thought']

							if self.speak_thoughts and tts is not None:
								arguments = {
									'text': thought,
									'stop_event': stop_event,
									'sound_stop_event': sound_stop_event,
									'tts': tts
								}
								t = threading.Thread(target=self.csp.tts, args=(arguments,))
								t.start()

							if command == "TaskComplete":
								task_complete_answer = command_argument
								break
							if command == "Ask":
								ask_question = command_argument
								break
							validity_output = response
							reasoning_context.append(self.ch.single_message_context("assistant", validity_output, False))
						except Exception as e:
							print("Error parsing JSON: "+str(e))

					if task_complete_answer:

						output = "1. Below is the ANSWER to the user's request. \n" 
						output += "2. Use the information below in your response. \n"
						output += "4. Only use this information to answer the user's question. No extraneous content like URLs, or repeating the content of other output, unless explicitly requested. \n"
						output += "5. Only provide the concise answer or a summary of what was done. No samples. \n"

						output += "\n\n"
						output += "Answer: "+str(task_complete_answer)

					if ask_question:

						output = "1. Please ask the user the following question. \n" 
						output += "2. Use the answer to determine what to do next. \n"
						output += "\n\n"
						output += "Question: "+str(ask_question)

					return output

	def check_for_task_completion(self, task, reasoning_context, stop_event):
		#Check for task completion
		print_text("Chaining (Completion Check): ", "yellow")
		completion_prompt = "Task: "+task
		completion_prompt += "Does the content of this conversation indicate the task is complete? (Yes/No)"
		reasoning_context_copy = reasoning_context.copy()
		reasoning_context_copy.append(self.ch.single_message_context("user", completion_prompt, False))
		#print(reasoning_context_copy)
		response = self.request(
			messages=reasoning_context_copy, 
			model="gpt-4", #Best at choosing tools
			stop_event=stop_event, 
			response_label=False,
			max_tokens=10
		)
		if "yes" in response.lower():
			reasoning_context_copy.append(self.ch.single_message_context("user", response, False))

			#Get completion reason
			print_text("Chaining (Completion Reasoning): ", "yellow")
			completion_reason_prompt = "In ONE sentence, respond to the task and describe what's been done."
			reasoning_context_copy.append(self.ch.single_message_context("user", completion_reason_prompt, False))
			#print(reasoning_context_copy)
			response = self.request(
				messages=reasoning_context_copy, 
				model="gpt-4", #Best at choosing tools
				stop_event=stop_event, 
				response_label=False,
				max_tokens=200
			)
			return response
		else:
			return None

	def request_boolean(self, question, stop_event=None, silent=True):
		prompt = "Answer the following with 'True' or 'False'\n\n"
		prompt += question
		message = [self.ch.single_message_context("user", prompt, False)]

		counter = 0

		while counter < 3:
			response = self.request(
				messages=message,
				model="gpt-4",
				stop_event=stop_event,
				response_label=False,
				temperature=0,
				silent=silent,
				max_tokens=10
			)
			if "true" in str(response.lower()):
				return True
			elif "false" in str(response.lower()):
				return False
			else:
				counter += 1

		return None

	def clarify_task(self, task, stop_event=None):
		prompt = "1. Respond with a more concise and clear task. Limit prose.\n"
		prompt += "2. Do not provide a solution, only state what needs to be done.\n"
		prompt += "3. If the task requires more than one logical step, respond with a shorter, clear, concise task for the first logical step. Limit prose.\n"
		prompt += "Task: "+task+"\n"
		message = [self.ch.single_message_context('system', prompt, False)]

		response = self.request(
			messages=message, 
			model="gpt-4", #Best at choosing tools
			stop_event=stop_event, 
			response_label=False,
			silent=True
		)

		return response
	
	def get_command_argument(self, task, command, description, argument, stop_event=None):
		#Get the argument
		prompt = "1. Respond with the necessary argument to accomplish the task.\n"
		prompt += "2. If the command is incorrect to accomplish the task, respond with 'Incorrect command'.\n"
		prompt += "3. Provide only the argument as specified in Argument'.\n"
		prompt += "Task: "+task+"\n"
		prompt += "Command: "+command+"\n"
		prompt += "Description: "+description+"\n"
		prompt += "Argument format: "+argument+"\n"

		#print_text(prompt)

		print_text("ARGUMENT ("+command+"): ", "green")
		message = [self.ch.single_message_context('system', prompt, False)]
		response = self.request(
				messages=message, 
				model="gpt-4", #Best at choosing tools
				stop_event=stop_event, 
				response_label=False
			)
		return response

	def validity_prompt(self, task, stop_event=None):
		prompt = "Task: "+task+"\n"
		prompt += "\n"
		prompt += "You are an AI that helps users complete tasks.\n"
		prompt += "\n"
		prompt += "Commands:\n\n"
		prompt += self.commh.get_command_info_text(self.commh.data)
		prompt += "Command: TaskComplete\n"
		prompt += "Description: Run this command if the task is complete and you are ready to return an answer to the user.\n"
		prompt += "Argument format: Answer/Reasoning (String). Limit prose.\n"
		prompt += "\n"
		prompt += "\n"
		prompt += "Rules:\n"
		prompt += "1. The context of this conversation is about accomplishing a task.\n"
		prompt += "2. Refer to messages previous in this conversation for necessary information and task status.\n"
		prompt += "4. Do not add any extra data to the JSON response than what is detailed in the example. No extraneous data.\n"
		prompt += "6. If the conversation is looping, Ask the user for more information.\n"
		prompt += "7. Be diligent about when the task is complete. If enough information has been gathered or the steps have been completed, run TaskComplete.\n"
		prompt += "8. Don't loop. If you are repeating yourself, or cannot find a solution, run TaskComplete\n"
		prompt += "\n"
		prompt += "Response Format: \n"
		prompt += "{\n"
		prompt += "    \"thoughts\": {\n"
		prompt += "        \"thought\": \"<what is your current thought to complete the task>\",\n"
		#prompt += "        \"criticism\": <based on what's already been done, what can be done differently?>\",\n"
		prompt += "        \"command\": \"<the command to accomplish the next step>\",\n"
		prompt += "        \"argument\": \"<the argument to the command>\",\n"
		prompt += "    } //Dont forget this bracket!\n"
		prompt += "}\n"

		return prompt
	
	def get_task_from_conversation(self, messages, stop_event):
		#Get the argument
		prompt = "1. The conversation below has earlier messages at the top, and the most recent message at the bottom.\n"
		prompt += "2. Respond with the task the user is trying to accomplish. Only prose.\n"
		prompt += "3. Do not try to accomplish the task here. Only recite what the task is.\n"
		prompt += "4. Include all necessary details and information to complete the task (URLs, numbers, etc).\n"
		prompt += "5. Do not answer the question or have a conversation.\n"
		prompt += "6. If the user and the AI are simply having a conversation or the latest message changes the subject, simply reply 'None'.\n"
		prompt += "\n"
		prompt += "Conversation:\n"
		#Get the last three messages and add them to the prompt
		last_three_messages = messages[-3:]
		for message in last_three_messages:
			prompt += message['role'].upper()+": "
			prompt += message['content']+"\n\n"
		prompt += "Task: "

		message = self.ch.single_message_context('user', prompt, False)

		response = self.request(
			messages=[message],
			stop_event=stop_event,
			#temperature = 0,
			response_label=False
			)

		if not response:
			logging.error("Daisy request error: No response")
			return None
		if response.startswith("None"):
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
				try:
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
				except ValueError as e:  # Handle the ValueError for each chunk
					if 'invalid literal for int() with base 16' in str(e):
						logging.error("stream_queue_sentences(): Error parsing a chunk of server response. Skipping this chunk and moving to the next one...")
						traceback.print_exc()
						continue  # Skip to the next chunk
					else:
						raise e
			if not silent:
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

