import socket
import sys
 
HOST,PORT = "192.168.31.69",19984
 
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT)) #连接边缘端

    sendData = str.encode('hello')
    sock.send(sendData)

    recvData = sock.recv(1024)
    print(bytes.decode(recvData))

    sendData = str.encode('hello2')
    sock.send(sendData)

    recvData = sock.recv(1024)
    print(bytes.decode(recvData))
    
    # print("Closing connection")
    # sock.close()
 
if __name__ == '__main__':
    main()