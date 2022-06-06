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
from edgeServerThread import edgeServerThread


class MainWiget(QWidget):
    def __init__(self):
        super().__init__()

        userServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ip_port=("192.168.31.69",19984)
        userServer.bind(ip_port)
        userServer.listen(100)
        print('listening for a client...')
        while True:
            # print('ABC for a client...')
            userClient, addr = userServer.accept()
            print(userClient, addr)
            userClient = edgeServerThread(userClient, addr) # 创建线程
            userClient.startScanSig.connect(self.scan) # 接收信号，执行扫描
            userClient.run() #执行线程

    def scan(self):
        print("开始扫描乐乐乐乐乐乐乐乐乐乐乐乐！")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWidget = MainWiget()
    sys.exit(app.exec_())