import yaml

"""
In the terminal run : 
conda env export > environment.yml

then run this code
"""

with open('environment.yml', 'r') as file:
    env = yaml.safe_load(file)

with open('requirements.txt', 'w') as file:
    for dependency in env['dependencies']:
        if isinstance(dependency, str):
            file.write(dependency + '\n')
        elif isinstance(dependency, dict) and 'pip' in dependency:
            for pip_dep in dependency['pip']:
                file.write(pip_dep + '\n')
