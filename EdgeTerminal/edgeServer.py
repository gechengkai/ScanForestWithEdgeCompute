import socketserver
import os
import sys
import time
import threading
 
ip_port=("192.168.31.69",19984)
 
class MyServer(socketserver.BaseRequestHandler):
    def handle(self):
        userClinet = self.request
        userAddr = self.client_address# IP地址
        print("Accepted connection from: ", userAddr) 
        while True:
            try:
                data = userClinet.recv(1024)
                if (bytes.decode(data) =='exit'):
                    break
                else:
                    print('Received data form UserTerminal:  ', bytes.decode(data))    
                    userClinet.send(str.encode('连接成功！')) #给用户端一个连接状态回复
            except Exception as e:
                print(e)
                break   
if __name__ == "__main__":
    s = socketserver.ThreadingTCPServer(ip_port, MyServer)
    print("start listen")
    s.serve_forever()