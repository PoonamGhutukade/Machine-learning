""" 31. Write a Python program to get system command output. """
import subprocess

print()
# It shows all list of file and directory
text = subprocess.check_output("dir", shell=True, universal_newlines=True)
print("dir command to list file and directory:")
print()
print(text)
