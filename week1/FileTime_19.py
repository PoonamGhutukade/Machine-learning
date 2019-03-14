""" 19. Write a Python program to get file creation and modification date/times. """

import os.path, time

# getmtime function for last modification
print("Last Modified : %s" % time.ctime(os.path.getmtime('/home/admin1/f1.txt')))

print("Creation Time: %s" % time.ctime(os.path.getctime('/home/admin1/f1.txt')))

# next line output in not in proper format
print(os.path.getmtime('/home/admin1/f1.txt'))