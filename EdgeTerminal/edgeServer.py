import socketserver
import os
import sys
import time
import threading
 
ip_port=("192.168.31.69",19984)
 
class MyServer(socketserver.BaseRequestHandler):
    def handle(self):
        print("conn is :",self.request) # 连接句柄
        print("addr is :",self.client_address) # IP地址
        while True:
            try:
                self.str = self.request.recv(8)
                data = bytearray(self.str)
                if (bytes.decode(data)!='exit'):
                    break
                else :
                    print('Received data form UserTerminal:  ', bytes.decode(data))    
                    self.request.send(str.encode('连接成功！')) #给用户端一个连接状态回复
            except Exception as e:
                print(e)
                break
 
 
if __name__ == "__main__":
    s = socketserver.ThreadingTCPServer(ip_port, MyServer)
    print("start listen")
    s.serve_forever()