import socket
import sys, requests, json, time, base64, os, logging, threading, argparse, os
from sys import platform
from lxml import etree
from PIL import Image
from PyQt5 import QtCore, QtGui, QtNetwork
from PyQt5.QtCore import QUrl, QByteArray, QThread, qDebug, pyqtSignal, QObject, Qt, pyqtSlot
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtWidgets import QApplication, QGroupBox, QToolBox, QVBoxLayout, QWidget, QPushButton, QLabel,QComboBox, QLineEdit, QGridLayout, QDesktopWidget, QHBoxLayout, QProgressBar, QStackedWidget
from PyQt5.QtGui import QFontDatabase, QPalette, QPixmap, QBrush, QFont, QIcon

from grpc import server
from userServerThread import userServerThread


PASS_SAVE_PATH = 'UserTerminal/passback/'

class UserMainWidget(QWidget):
    def __init__(self):
        super().__init__()
        # 创建文件夹
        self.mkdir(PASS_SAVE_PATH)
    #创建文件夹
    def mkdir(self, path):
        # 去除首位
        path = path.strip()
        # 去除尾部 \ 符号
        path = path.rstrip("\\")
        # 判断路径是否存在
        # 存在     True
        # 不存在   False
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)
            print(path + ' 创建成功')
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print(path + ' 目录已存在')
            return False    
    def main(self):
        userServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ip_port=("192.168.31.53",5150)
        userServer.bind(ip_port)
        userServer.listen(100)
        print('listening for a client...')
        while True:
            # print('ABC for a client...')
            edgeClient, addr = userServer.accept()
            print(edgeClient, addr)
            edgeClient = userServerThread(edgeClient, addr)
            edgeClient.run()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    userMainWidget = UserMainWidget()
    userMainWidget.main()
    sys.exit(app.exec_())