import socket
import sys
 
HOST,PORT = "192.168.31.69",19984
 
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT)) #连接边缘端
    
    data = sock.recv(8)
    print(bytes.decode(data))

    data = str.encode('exit')
    sock.send(data)
    
    print("Closing connection")
    sock.close()
 
if __name__ == '__main__':
    main()