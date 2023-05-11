import subprocess

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, _ = process.communicate()
    return output.decode().strip()

# Uninstall 'daisy_llm' package
run_command('echo y | pip uninstall daisy_llm')

# Install the current directory as a package
run_command('pip install c:/Users/myrak/Documents/GitHub/daisy_llm_tools/')
