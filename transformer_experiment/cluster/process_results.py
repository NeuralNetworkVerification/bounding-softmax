import os
import sys

i = 1
with open("benchmark_set_all", 'r') as in_file:
    for line in in_file.readlines():
        s = line.split()
        result = open(f"history/slurm-{i}.log", 'r').read().split("\n")
        if "unsat" in result:
            print(",".join(s) + ",unsat")
        elif "misclassify!" in result:
            print(",".join(s) + ",misclassify")
        else:
            print(",".join(s) + ",unknown")
        i += 1
