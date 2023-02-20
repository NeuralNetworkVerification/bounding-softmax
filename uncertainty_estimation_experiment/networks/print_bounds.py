import pickle

with open("./bounds/bounds_ind0_eps0.04.pickle", 'rb') as fp:
    bounds = pickle.load(fp)

lbs = bounds["lbs"]
ubs = bounds["ubs"]

for layer in range(len(lbs)):
    print(f"layer {layer}:")
    print(lbs[layer])
    print(ubs[layer])
