import socket
import sys
from turtle import st
 
HOST,PORT = "192.168.31.69",19984

# 连接
def connectEdge():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT)) #连接边缘端

    sendData = str.encode('connct')
    sock.send(sendData)

    recvData = sock.recv(1024)
    print(bytes.decode(recvData))
    
    print("Closing connection")
    sock.close()

# 开始扫描
def startScan():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT)) #连接边缘端

    sendData = str.encode('scan')
    sock.send(sendData)

    recvData = sock.recv(1024)
    print(bytes.decode(recvData))
    
    print("Closing connection")
    sock.close()
 
if __name__ == '__main__':
    # connectEdge()
    startScan()