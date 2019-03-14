""" 12. Write a python program to call an external command in Python. """

from subprocess import call
import os

"""call() function is used to call external command, 
ls -l command list all the files in long format. 
It display subroutines, permissions, modification time,owner etc
"""
call(["ls", "-l"])

print("-----------------------OR-------------------------")


print(os.system('ls -l'))
