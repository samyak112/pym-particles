test = {}

print('test')

with open("test.txt", "rb") as f:
    while chunk := f.read(512):
        # process chunk
        if chunk in test:
            test[chunk] += 1
        else:
            test[chunk] = 1

y = 0

print(len(test))

for key,value in test.items():
    if(value > 1):
        print(value)
