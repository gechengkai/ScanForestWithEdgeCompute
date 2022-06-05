import socketserver
import os
import sys
import time
import threading
 
ip_port=("192.168.31.53",5150)
 
class UserServer(socketserver.BaseRequestHandler):
    def handle(self):
        print("conn is :",self.request) # 连接句柄
        print("addr is :",self.client_address) # IP地址
        while True:
            try:
                self.str = self.request.recv(8)
                data = bytearray(self.str)
                headIndex = data.find(b'\xff\xaa\xff\xaa')
                print(headIndex)
                 
                if headIndex == 0:
                    allLen = int.from_bytes(data[headIndex+4:headIndex+8], byteorder='little')
                    print("len is ", allLen)
  
                    curSize = 0
                    allData = b''
                    while curSize < allLen:
                        data = self.request.recv(1024)
                        allData += data
                        curSize += len(data)
  
                    print("recv data len is ", len(allData))
                    #接收到的数据，前64字节是guid，后面的是图片数据
                    arrGuid = allData[0:64]
                    #去除guid末尾的0
                    tail = arrGuid.find(b'\x00')
                    arrGuid = arrGuid[0:tail]
                    strGuid = str(int.from_bytes(arrGuid, byteorder = 'little')) #for test
                     
                    print("-------------request guid is ", strGuid)
                    imgData = allData[64:]
                    strImgFile = "2.jpg"
                    print("img file name is ", strImgFile)
  
                    #将图片数据保存到本地文件
                    with open(strImgFile, 'wb') as f:
                        f.write(imgData)
                        f.close()
                         
                    break
            except Exception as e:
                print(e)
                break
 
 
if __name__ == "__main__":
    s = socketserver.ThreadingTCPServer(ip_port, UserServer)
    print("start listen")
    s.serve_forever()