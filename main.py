import threading

import src.daisy_llm.DaisyCore as module_loader
import src.daisy_llm.ContextHandlers as context_handlers

ch = context_handlers('daisy.db')

#Instantiate ModuleLoader and ContextHandlers for global use by front-ends
ml = module_loader(ch, "modules", "configs.yaml")
update_modules_loop_thread = threading.Thread(target=ml.update_modules_loop)
update_modules_loop_thread.start()

ml.process_main_start_instances()