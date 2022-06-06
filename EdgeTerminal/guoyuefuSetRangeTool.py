import os
from PIL import Image
import argparse
import threading
from sys import platform
import sys, requests, json, time, base64, os, socket
from PyQt5 import QtCore, QtGui, QtNetwork
from PyQt5.QtCore import QUrl, QByteArray, QThread
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtWidgets import QApplication, QTabWidget, QWidget, QPushButton, QLabel,QComboBox, QLineEdit, QGridLayout, QDesktopWidget, QHBoxLayout, QProgressBar, QStackedWidget
from PyQt5.QtCore import pyqtSignal, QObject, Qt, pyqtSlot
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QFont


class TaianSetTool(QWidget):
    def __init__(self):
        super().__init__() # 初始化
        self.initUI()  # 初始化界面
        """泰安相关全局变量"""
        self.positionList = []
        self.backgroundRole()
        self.ta_savePath = "detectDir\\taian\\capImg\\"
        #云台扫描范围
        #self.setRangeOfCamera()
        # self.testSetRange()
        # self.mkdir(self.ta_savePath)

    def initUI(self):

        self.scanTab = QWidget()
        self.recTab = QWidget()
        self.histTab = QWidget()

        
        #------------------#
        #   位置复现窗口
        #------------------#
        # topWidget
        self.rec_imageLabel = QLabel('请输入参数并点击\n查询按钮')
        self.rec_imageLabel.setWordWrap(True)
        self.rec_imageLabel.setAlignment(Qt.AlignCenter)#字体居中
        # self.rec_imageLabel.setStyleSheet("color:white")#字体颜色
        # self.rec_imageLabel.setFont(QFont("Microsoft YaHei",25,QFont.Bold))#设置字体  Roman times
        self.rec_imageLabel.setFixedSize(640,370)
        self.rec_imageLabel.setScaledContents (True)
        self.rec_imageLabel.setObjectName('rec_imageLabel') 
        self.rec_imageLabel.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))  # 设置字体  Roman times


        self.rec_topWidget = QWidget()
        self.rec_topLayout = QGridLayout()
        self.rec_topLayout.addWidget(self.rec_imageLabel,0,0)
        #self.rec_topLayout.addWidget(self.rec_prgBar,1,0)
        self.rec_topWidget.setLayout(self.rec_topLayout)
        # topWidget 大小
        self.rec_topWidget.setFixedSize(670,420)
        self.rec_topWidget.setObjectName('rec_topWidget') 
        #middleWidget
        self.rec_panLabel = QLabel('水平值')
        self.rec_panLabel.setFixedSize(60,28)
        # self.rec_panLabel.setStyleSheet("color:white")
        self.rec_panLabel.setAlignment(Qt.AlignRight)

        self.rec_panLineEdit = QLineEdit('0')
        self.rec_panLineEdit.selectAll()
        self.rec_panLineEdit.setFocus()

        self.rec_tiltLabel = QLabel('俯仰值')
        self.rec_tiltLabel.setFixedSize(60,28)
        self.rec_tiltLabel.setAlignment(Qt.AlignRight)#文字居右对齐

        self.rec_tiltLineEdit = QLineEdit('0')
        self.rec_tiltLineEdit.selectAll()
        self.rec_tiltLineEdit.setFocus()

        self.rec_zoomLabel = QLabel('变焦值')
        self.rec_zoomLabel.setFixedSize(60,28)
        self.rec_zoomLabel.setAlignment(Qt.AlignRight)

        self.rec_zoomLineEdit = QLineEdit('0')
        self.rec_zoomLineEdit.selectAll()
        self.rec_zoomLineEdit.setFocus()

        self.rec_paramLayout = QGridLayout()
        self.rec_paramLayout.addWidget(self.rec_panLabel, 0, 0)
        self.rec_paramLayout.addWidget(self.rec_panLineEdit, 0, 1)
        self.rec_paramLayout.addWidget(self.rec_tiltLabel, 0, 2)
        self.rec_paramLayout.addWidget(self.rec_tiltLineEdit, 0, 3)
        self.rec_paramLayout.addWidget(self.rec_zoomLabel, 0, 4)
        self.rec_paramLayout.addWidget(self.rec_zoomLineEdit, 0, 5)

        self.rec_paramWidget = QWidget()
        self.rec_paramWidget.setLayout(self.rec_paramLayout)
        self.rec_paramWidget.setFixedSize(500,50)

        self.rec_middleWidget = QWidget()
        self.rec_middleLayout = QGridLayout()
        self.rec_middleLayout.addWidget(self.rec_paramWidget,0,0)
        self.rec_middleWidget.setLayout(self.rec_middleLayout)
        self.rec_middleWidget.setFixedSize(670,65)
        self.rec_middleWidget.setObjectName('rec_middleWidget')

        #bottomWidget
        # 设置按钮
        self.rec_askBtn = QPushButton('查询位置')
        self.rec_askBtn.clicked.connect(self.getCameraPositon)
        self.rec_askBtn.setStyleSheet("color:green")
        # self.rec_askBtn.setFixedSize(670,35)
        # 查询按钮
        self.rec_setBtn = QPushButton('设置位置')
        self.rec_setBtn.clicked.connect(self.recept)
        self.rec_setBtn.setStyleSheet('color:red')
        # self.rec_setBtn.setFixedSize(670,35)

        
        self.rec_btnWidget = QWidget()
        self.rec_btnLayout = QGridLayout()

        self.rec_btnLayout.addWidget(self.rec_askBtn, 1,1)
        self.rec_btnLayout.addWidget(self.rec_setBtn, 1,0)
        self.rec_btnWidget.setLayout(self.rec_btnLayout)
        self.rec_btnLayout.setSpacing(0)
        self.rec_btnWidget.setFixedSize(670,50)

        self.rec_bottomWidget = QWidget()
        self.rec_bottomLayout = QGridLayout()
        self.rec_bottomLayout.addWidget(self.rec_btnWidget,0,0)
        self.rec_bottomWidget.setLayout(self.rec_bottomLayout)
        self.rec_bottomLayout.setSpacing(0)
        self.rec_bottomWidget.setFixedSize(670,60)
        self.rec_bottomWidget.setObjectName('rec_bottomWidget')

        # recWidget界面布局
        #recWidget
        self.recLayout = QGridLayout()
        self.recLayout.addWidget(self.rec_topWidget, 0, 0)
        self.recLayout.addWidget(self.rec_middleWidget, 1, 0)
        self.recLayout.addWidget(self.rec_btnWidget, 2, 0)
        self.recLayout.setSpacing(0)

        self.recTab.setLayout(self.recLayout)

        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.recTab)
        self.setLayout(self.mainLayout)

        # with open('UI\\taian\\source.qss') as f:
        #     qss = f.read()
        #     print(qss)
        #     self.setStyleSheet(qss)


    '''**********************************************************************************************'''
    #get请求获取token
    def getTokenRequest(self):
        #设置超时时间 10s
        socket.setdefaulttimeout(10)
        #header 信息
        headers = {'Authorization': 'bm9uZ2RhMTpub25nZGFAMTIz'}
        #get请求
        tokenUrl = "http://218.201.180.118:9840/apiserver/v1/user/authentication-token"
        try:
            tokenResponse = requests.get(tokenUrl, headers=headers,timeout=(3,7))
        except:
            print("------------------------------token获取失败")
            tokenResponse = None
        if tokenResponse == None:
           return None
        #如果token不为空
        else:
            #从返回值中获取token，保存到全局变量token中
            data = json.loads(tokenResponse.text)
            token = data.get("data")
            if token == None :
                print("Error 获取token失败！")
                return None
            else:
                print("获取Token成功！" + ":" + token)
                return token
        
        


    #设置云台位置
    def setPositonRequest(self, positionStr):
        #获取token
        token = self.getTokenRequest()
        if token != None:
            #取出水平、俯仰、放大值
            tmpList = str(positionStr).split("#")
            panStr = tmpList[0]
            tiltStr = tmpList[1]
            zoomStr = tmpList[2]
            setPosiStr = panStr + "  "+tiltStr + "  "+zoomStr
            print("panStr:%s"%panStr)
            print("tiltStr:%s"%tiltStr)
            print("zoomStr:%s"%zoomStr)

            #组成json
            postData = dict(camera_id="37090200001311000980", pan = int(panStr), tilt=int(tiltStr), zoom=int(zoomStr))
            setPostionUrl = "http://218.201.180.118:9840/apiserver/v1//device/ptz/postion?token="+token
            #print(setPostionUrl)
            try:
                setPostionResponds = requests.post(setPostionUrl, json=postData,timeout=(3,7))
            except:
                print('1----------------------------------设置云台位置失败')
                self.rec_imageLabel.setText('设置位置失败，请重试!')
                setPostionResponds = None
            if setPostionResponds != None:
                print(setPostionResponds.text)
                data = json.loads(setPostionResponds.text)
                setPositionRes = data.get("data")
                self.rec_imageLabel.setText('设置位置成功!\n'+setPosiStr)
                return setPositionRes 
            else:
                print('2----------------------------------设置云台位置失败')
                self.rec_imageLabel.setText('设置位置失败，请重试!')
                return None
            

            
        else:
            print('0------------------------------设置云台位置时，获取token失败')
            self.rec_imageLabel.setText('设置位置失败，请重试!')
            return None




    #获取当前摄像头位置
    def getCameraPositon(self):
        #设置超时时间 10s
        socket.setdefaulttimeout(10)
        token = self.getTokenRequest()# 获取token
        if token != None:

            frontStr = "http://218.201.180.118:9840/apiserver/v1/device/ptz/postion?token="
            backStr = "&camera_id=37090200001311000980"

            cptureUrl = frontStr + token + backStr
            try:
                cptureResponse = requests.get(cptureUrl,timeout=(3,7))
            except:
                print("-------------------------------获取当前摄像机位置失败")
                cptureResponse = None
            if cptureResponse != None:
                tmpStr1 = json.loads(cptureResponse.text)
                dataDic = tmpStr1.get("data")#取出data
                if dataDic == None:
                    print("data信息为空：dataDic")
                    print(dataDic)
                    # tmpStr = self.positionList[self.num]
                    # #取出水平、俯仰、放大值
                    # tmpList = tmpStr.split("#")
                    # panStr = tmpList[0]
                    # tiltStr = tmpList[1]
                    # zoomStr = tmpList[2]
                    # camPosiStr = panStr+"#"+tiltStr+"#"+zoomStr
                    return None
                
                else:
                    pan = str(dataDic["pan"])#取出data中的pic_buf
                    tilt = str(dataDic["tilt"])#取出data中的pic_buf
                    zoom = str(dataDic["zoom"])#取出data中的pic_buf
                    self.rec_panLineEdit.setText(pan)
                    self.rec_tiltLineEdit.setText(tilt)
                    self.rec_zoomLineEdit.setText(zoom)
                    camPosiStr = pan + "  "+tilt + "  "+zoom
                    # return camPosiStr
                    self.rec_imageLabel.setText('查询成功!\n'+camPosiStr)
            
            else:
                print("------------------------------获取当前摄像机位置失败")
                self.rec_imageLabel.setText('查询失败!')
                return None
        else:
            print('------------------------------获取当前摄像头位置过程中，获取token失败')
            self.rec_imageLabel.setText('查询失败!')
            return None  

    #泰安画面复现
    def recept(self):
        positionList = []
        #获取pan，tilt，zoom
        pan = self.rec_panLineEdit.text()
        tilt = self.rec_tiltLineEdit.text()
        zoom = self.rec_zoomLineEdit.text()
        posiStr = pan+"#"+tilt+"#"+zoom
        positionList.append(posiStr)
        # 更改Label提示
        self.rec_imageLabel.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))  # 设置字体  Roman times
        self.rec_imageLabel.setText('准备中，请等待...')
        print("positionList:%s"%positionList)
        self.setPositonRequest(posiStr)

    '''*******************************************************************************************'''




if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = TaianSetTool()
    w.show()
    sys.exit(app.exec_())