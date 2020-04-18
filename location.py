fh = open('C:/Users/LENOVO/Desktop/location2.txt', 'rb')
for line in fh.readlines():
    print(line[-28:-1])
