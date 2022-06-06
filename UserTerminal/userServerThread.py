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

class userServerThread(QThread):
    def __init__(self, edgeClient, addr):
        print('有新连接建立！')
        super().__init__()
        self.edgeClient, self.addr = edgeClient, addr
    def run(self) -> None:
        # print('有新连接建立！')
        while True:
            try:
                self.str = self.edgeClient.recv(8)
                data = bytearray(self.str)
                headIndex = data.find(b'\xff\xaa\xff\xaa')
                print(headIndex)
                 
                if headIndex == 0:
                    allLen = int.from_bytes(data[headIndex+4:headIndex+8], byteorder='little')
                    print("len is ", allLen)
  
                    curSize = 0
                    allData = b''
                    while curSize < allLen:
                        data = self.edgeClient.recv(1024)
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

                    #接收到的数据，接下来64字节是name，后面的是图片数据
                    arrName = allData[64:128]
                    #去除末尾的0
                    tail = arrName.find(b'n')
                    print('tail :',tail)
                    arrName = arrName[0:tail]
                    strName = arrName.decode() #
                    
                    print("-------------request Name is ", strName)
                    imgData = allData[128:]
                    strImgFile = strName+".jpg"
                    print("img file name is ", strImgFile)
  
                    #将图片数据保存到本地文件
                    with open(strImgFile, 'wb') as f:
                        f.write(imgData)
                        f.close()
                         
                    break
            except Exception as e:
                print(e)
                break

        self.edgeClient.close()