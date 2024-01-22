import threading

from daisy_llm import DaisyCore as module_loader
from daisy_llm import ContextHandlers as context_handlers

ch = context_handlers("daisy.db")

# Instantiate ModuleLoader and ContextHandlers for global use by front-ends
ml = module_loader(ch, "modules", "configs.yaml")
update_modules_loop_thread = threading.Thread(target=ml.update_modules_loop)
update_modules_loop_thread.start()

ml.process_main_start_instances()
