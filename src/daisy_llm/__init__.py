# Import the necessary modules and classes that define the package
from .SoundManager import SoundManager
from .ChatSpeechProcessor import ChatSpeechProcessor

# from .ModuleLoader import ModuleLoader
from .chat import Chat
from .context_handlers import ContextHandlers
from .ConnectionStatus import ConnectionStatus
from .LoadTts import LoadTts
from .CommandHandlers import CommandHandlers
from .DaisyCore import ModuleLoader as DaisyCore

# Define the package metadata
__name__ = "daisy_llm"
__version__ = "0.0.4"
__author__ = "Myra Krusemark"
__email__ = "daisy_llm_tools@myrakrusemark.com"

# Export the classes and metadata for use by other modules and packages
__all__ = [
    "chat",
    "ContextHandlers",
    "DaisyCore",
    "ChatSpeechProcessor",
    "SoundManager",
    "ConnectionStatus",
    "LoadTts",
    "CommandHandlers",
    "__name__",
    "__version__",
    "__author__",
    "__email__",
]
