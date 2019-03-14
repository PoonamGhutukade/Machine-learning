""" 32. Write a Python program to get the effective group id, effective user id, real group id,
a list of supplemental group ids associated with the current process.
Note: Availability: Unix."""

import os

os.initgroups("user", 2)
print("\n Effective Group Id:", os.getegid())
print("Effective UserId:", os.geteuid())

print("\n Real Group Id:", os.getgid())
print("Real User Id:", os.getuid())

print("\n List of supplemental Group Id:", os.getgroups())

print("-------------------------Program: 33---------------------------")
print("User Environment:", os.environ)
