
netpertToConfigToNum = dict()

i = 0
s = ""
with open("output.csv") as in_file:
    for line in in_file.readlines():
        net,eps,bound,num = line.split(",")
        if i > 0 and i % 7 == 0:
            print(s[:-1])
            s = f"{net},{eps},"
        s += bound + ","
        s += str(int(num)/ 5) + ","
        i += 1
        
