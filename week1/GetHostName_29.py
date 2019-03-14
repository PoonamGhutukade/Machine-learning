""" 29. Write a Python program to get the name of the host on which the routine is running.
"""

import socket
# get hostname
hostN = socket.gethostname()

# get IP address
ipaddr = socket.gethostbyaddr(hostN)

print("Host Name:", hostN)
print("Ip addr: ", ipaddr)
