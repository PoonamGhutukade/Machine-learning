""" 23. Write a Python program to find the available built-in modules. """

import sys
import textwrap

module_name = ', '.join(sorted(sys.builtin_module_names))

# Display all modules in one line only if we use following line
# print(module_name)
print()
# default value for textwrap is 70
print(textwrap.fill(module_name, 90))