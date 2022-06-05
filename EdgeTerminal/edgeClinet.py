import socket
import sys
from pathlib import Path
from unicodedata import name
import glob, os

from numpy import dtype
 
HOST,PORT = "192.168.31.53",5150
 
def main():
    
    RES_SAVE_PATH = 'G:/ScanForestProgram-old/saoshan/detectDir/taian/detect/20210820-1554'

    picPaths = glob.glob(os.path.join(RES_SAVE_PATH, '*.jpg')) #获取图片路径
    # print(f'img_path list:{picPaths}')
    for picPath in picPaths:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        #包头标志
        arrBuf = bytearray(b'\xff\xaa\xff\xaa')
        
        #以二进制方式读取图片
        picData = open(picPath, 'rb')
        picBytes = picData.read()
        
        #图片大小
        picSize = len(picBytes)

        #图片名称
        picName = Path(picPath).stem #获取不带路径与扩展的文件名 
        print(f'Picture baseName: {picName}')
        picName = picName.ljust(64, "n") #补足64位
        nameBytes = bytearray(picName.encode())
        
        #数据体长度 = guid大小(固定) + 图片大小
        datalen = 64 + 64 + picSize
        
        #组合数据包
        arrBuf += bytearray(datalen.to_bytes(4, byteorder='little'))
        guid = 23458283482894382928948
        arrBuf += bytearray(guid.to_bytes(64, byteorder='little'))
        arrBuf += nameBytes
        arrBuf += picBytes
        


        sock.sendall(arrBuf)
        sock.close()
 
if __name__ == '__main__':
    main()