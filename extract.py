
f = open("b.txt")
a=open('extracted.txt',mode='w')
line = f.readline()  # 调用文件的 readline()方法
while line:
    print(line)  # 后面跟 ',' 将忽略换行符
    if line.find('/step')!=-1:
        a.write(line)
    line = f.readline()
a.close()
f.close()