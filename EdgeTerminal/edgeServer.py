from mimetypes import init
import socketserver
import os
import sys
import time
import threading
 
 
class ClientServer(socketserver.BaseRequestHandler):
    def handle(self):
        userClient = self.request
        userAddr = self.client_address# IP地址
        print("Accepted connection from: ", userAddr) 
        while True:
            try:
                data = userClient.recv(1024)
                if (bytes.decode(data) =='connect'):
                    print("CONNECT ORDER!")
                    userClient.send(str.encode('连接成功！')) #给用户端一个连接状态回复
                    break
                elif (bytes.decode(data) =='scan'):
                    print("SCAN ORDER!")
                    userClient.send(str.encode('开始扫描！')) #给用户端一个连接状态回复
                    break
                else:
                    print('Received data form UserTerminal:  ', bytes.decode(data))    
                    userClient.send(str.encode('收到！')) #给用户端一个连接状态回复
                    break
            except Exception as e:
                print(e)
                break   
        userClient.close()
        print("Close connection of :",userClient)

        
if __name__ == "__main__":
    ip_port=("192.168.31.69",19984)
    s = socketserver.ThreadingTCPServer(ip_port, ClientServer)
    print("start listen")
    s.serve_forever()