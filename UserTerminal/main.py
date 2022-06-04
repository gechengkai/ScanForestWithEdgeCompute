import socket


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostbyname('localhost') #自动获取本机地址
port = 5150

server.connect(('192.168.31.69',port))
data = server.recv(1024) # 指定保存接收数据的缓存区大小
print(bytes.decode(data))
while True:
    data = input('Enter text to send:')
    server.send(str.encode(data))
    data = server.recv(1024)
    print('Received from server:', bytes.decode(data))
    if (bytes.decode(data) == 'exit'):
        break
print("Closing connection")
server.close()