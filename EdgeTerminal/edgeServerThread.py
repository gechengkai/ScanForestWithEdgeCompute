#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:    :用户端多线程服务器
@Date            :2022/06/06 
@Author          :gechengkai
@version         :v1.1
'''
import os, sys, requests, json, time, base64, os, socket, urllib3.request, glob, argparse, threading, logging
from requests.api import head
from PIL import Image
# from yolo import YOLO
from sys import platform
from PyQt5 import QtCore, QtGui, QtNetwork
from PyQt5.QtCore import QUrl, QByteArray, QThread
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QGridLayout, QDesktopWidget, QHBoxLayout, QProgressBar, QStackedWidget
from PyQt5.QtCore import pyqtSignal, QObject, Qt, pyqtSlot
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QFont

from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

class edgeServerThread(QThread):
    # *******信号********
    # 开始扫描
    startScanSig = QtCore.pyqtSignal()

    def __init__(self, userClient, addr):
        print('有新连接建立！')
        super().__init__()
        self.userClient, self.addr = userClient, addr
    def run(self) -> None:
        # print('有新连接建立！')
        while True:
            try:
                data = self.userClient.recv(1024)
                if (bytes.decode(data) =='connect'):
                    print("CONNECT ORDER!")
                    self.userClient.send(str.encode('连接成功！')) #给用户端一个连接状态回复
                    break
                elif (bytes.decode(data) =='scan'):
                    print("SCAN ORDER!")
                    self.userClient.send(str.encode('开始扫描！')) #给用户端一个连接状态回复
                    self.startScanSig.emit() # 发送信号
                    break
                else:
                    print('Received data form UserTerminal:  ', bytes.decode(data))    
                    self.userClient.send(str.encode('收到！')) #给用户端一个连接状态回复
                    break
            except Exception as e:
                print(e)
                break   
            except Exception as e:
                print(e)
                break
        self.userClient.close()