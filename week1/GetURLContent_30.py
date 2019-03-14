""" 30. Write a Python program to access and print a URL's content to the console.

"""

from http.client import HTTPConnection
# It works for python3
from urllib.request import urlopen

print()
conn = HTTPConnection("example.com")
conn.request("GET", "/")
result = conn.getresponse()
# retrieve the entire contents
contents = result.read()
print("URL Contents for example.com: ", contents)

print("--------------------------------------------------------")
print()

# conn=HTTPConnection("http://Google.com/") this gives error as invalid URL
conn = HTTPConnection("Google.com")
conn.request("GET", "/")
result = conn.getresponse()
contents = result.read()
print("URL Contents for google.com: ", contents)
print()

print("------------------------------OR-----------------------")


# f = urllib.request.urlopen("http://Google.com/")  ---it works but we have to import urlopen in additional
f = urlopen("http://Google.com/")
myfile = f.read()
print("URL Contents for http://Google.com: ", myfile)
