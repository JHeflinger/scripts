"""
author: Jason Heflinger
description: adds static keyword to places that need it
"""

import sys

if (len(sys.argv) != 2):
    print("Wrong usage detected. Please provide just the path to a list of replacements formatted as <filepath> line")
    exit(1)

with open(sys.argv[1], "r") as file:
    content = file.readlines()
    for line in content:
        path = line.split("\"")[1]
        line = int(line[:-1].split(",")[1])
        edited = []
        with open(path, "r") as buf:
            edited = buf.readlines()
            edited[line - 1] = "static " + edited[line - 1]
        with open(path, "w") as buf:
            buf.writelines(edited);
    print("Finished " + str(len(content)) + " insertions")
