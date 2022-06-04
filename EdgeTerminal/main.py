import socket


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #创建套接字： ipv4 , TCP连接
host = '192.168.31.69'
port = 5150
server.bind((host, port))# 绑定地址与端口
server.listen(5)# 监听
print('Listening for a client....')
clinet, addr = server.accept()# 获取客户端句柄和地址
print('Accept connection from:', addr)
clinet.send(str.encode('Welcom to my server!'))# 向客户端发送消息
while True:
    data = clinet.recv(1024)# 指定存储接收数据的缓冲区的大小
    if (bytes.decode(data) == 'exit'):
        break
    else:
        print('Received data from client:',bytes.decode(data))
        clinet.send(data)
print('Ending the connection')
clinet.send(str.encode('exit'))
clinet.close()