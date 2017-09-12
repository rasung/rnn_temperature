import random
import csv
random.random()

GOOD = 0
BAD = 1     


makeDataPath = 'data.csv'

f = open(makeDataPath, 'w', encoding='utf-8', newline='')

wr = csv.writer(f)


for i in range(1000000):
    
    data = []
    sum = 0;
    for j in range(10):
        value = random.randrange(50,120)
        data.append(value)
        sum += value

    avr = 0;
    for j in range(10):
        avr = sum/10

    if avr < 100:
        data.append(GOOD)
    else:
        data.append(BAD)

    wr.writerow(data)

