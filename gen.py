import random
import sys
size = int(sys.argv[1])

for i in range(1,size+1):
    for j in range(1,size+1):
        if i == j:
            pass
        else:
            a = random.randint(1, 100)
            if a > 90:
                print i, j