import logging
from src.daisy_llm_myrakrusemark.Logging import Logging

logger = Logging('daisy.log')
logger.set_up_logging()

import os
import sys
import threading
import time
import concurrent.futures

import src.daisy_llm_myrakrusemark.ModuleLoader as module_loader
import src.daisy_llm_myrakrusemark.ContextHandlers as context_handlers
from src.daisy_llm_myrakrusemark.Logging import Logging

ch = context_handlers('daisy.db')

#Instantiate ModuleLoader and ContextHandlers for global use by front-ends
ml = module_loader(ch, "modules", "configs.yaml")
update_modules_loop_thread = threading.Thread(target=ml.update_modules_loop)
update_modules_loop_thread.start()

ml.process_main_start_instances()