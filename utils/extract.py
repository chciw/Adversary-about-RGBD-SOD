
f = open("input.txt")
a=open('extracted.txt',mode='w')
line = f.readline()  
while line:
    print(line)  # 后面跟 ',' 将忽略换行符
    if line.find('/step')!=-1:
        a.write(line)
    line = f.readline()
a.close()
f.close()
