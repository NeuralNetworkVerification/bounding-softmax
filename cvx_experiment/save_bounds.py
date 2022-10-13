import sys
import numpy as np
import os
import pickle

for filename in os.listdir("./bounds_new/"):
    if ".pickle" not in filename:
        lbs = []
        ubs = []
        print(filename)
        with open(f'./bounds_new/{filename}', 'r') as in_file:
            for line in in_file.readlines():
                if "Layer 10:\n" == line:
                    break
                if "Layer " in line:
                    lbs.append([])
                    ubs.append([])
                    continue
                if "LB" in line and "UB" in line:
                    l = line.strip().split()
                    lb = float(l[2][:-1])
                    ub = float(l[4])
                    lbs[-1].append(lb)
                    ubs[-1].append(ub)
        print(filename)
        while lbs[0] == []:
            lbs = lbs[1:]

        while ubs[0] == []:
            ubs = ubs[1:]

        with open(f"./bounds/{filename}.pickle", "wb") as fp:
            pickle.dump({"lbs":lbs, "ubs": ubs}, fp)
