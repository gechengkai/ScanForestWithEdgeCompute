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


class CapturePicWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        # 信号
        # 抓图完毕信号
        captureImgFinishSig = QtCore.pyqtSignal()

        # 创建一个logger 
        self.logger = logging.getLogger("guoyuefu_capture") # 初始化logger
        self.logger.setLevel(logging.INFO) # Log 等级总开关
        # 创建一个handler，用于写入日志文件
        handler_file = logging.FileHandler(filename="guoyuefu.log", mode='a', encoding='utf-8')
        handler_file.setLevel(logging.INFO) #输出到 file 的 Log 等级
        # 创建一个handler，用来把日志写到cmd上
        handler_cmd = logging.StreamHandler()
        handler_cmd.setLevel(logging.INFO)
        # 设置日志格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - %(message)s")
        handler_file.setFormatter(formatter)
        handler_cmd.setFormatter(formatter)
        # 将相应的handler添加到logger对象中
        self.logger.addHandler(handler_file)

        # 全局变量
        self.capSavePath = "EdgeTerminal\\detectDir\\taian\\capImg\\" # where are these images that were cptured to store
        self.positionList = [] # 抓取图片摄像机位置参数列表
        self.deteSavePath = "EdgeTerminal\\detectDir\\taian\\detect\\" # 
        self.num = 0 # 定位positionList
        self.mkdir(self.capSavePath)
        self.mkdir(self.deteSavePath)

        #设置云台扫描范围
    def setRangeOfCamera(self):
        with open("UI\\taian\\guoyuefu\\guoyuefu.range") as lines:
            for line in lines:
                line = line.strip('\n')#去掉换行符
                rangeList = line.split('\t')#按空格分割字符串
                start = rangeList[0]
                end = rangeList[1]
                tilt = rangeList[2] 
                for i in range(int(start), int(end), 6):
                    tmpStr = str(i) + '-' + str(tilt) + '-' + "530"
                    self.positionList.append(tmpStr)
        print(self.positionList)

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

        
    def startScan(self):
        self.logger.info("开始扫描le ————————————>")
        self.startWork("init")

     #开始
    def startWork(self,flag):
        if flag == "init":
            # self.startCaptureSig.emit(len(self.positionList))      
            for i in range(25712,len(self.positionList)):
                self.num  = i
                self.logger.info("\n\n————————————————————————————————  %d  ————————————————————————————————"%int(self.num))
                self.logger.info("work:国悦府-第%d张图片" %(i))
                self.logger.info("work:国悦府-共%d张图片" %(len(self.positionList)))
                # 设定位置
                self.logger.info("work:1 设定云台位置")
                setPositionRes = self.setPositonRequest(self.positionList[self.num]) 
                if setPositionRes != None:
                    #延时3秒
                    time.sleep(3)
                    #截图
                    self.logger.info("work:2 截图")
                    imageBase64Str = self.cptureRequests()
                    while imageBase64Str == None:
                        self.logger.info("work:------------------------------未获取到图片,3秒后重试")
                        time.sleep(3)
                        imageBase64Str = self.cptureRequests()
                    #转成图片
                    imgdata = self.base64ToImage(imageBase64Str)
                    #保存
                    saveImagePath = self.saveCapImg(imgdata)
                    #发送展示抓取图像信号
                    # self.getImageSig.emit(saveImagePath)
                    #获取相机位置
                    self.logger.info("work:3 获取云台位置")
                    camPosStr = self.getCameraPositon()
                    while camPosStr == None:
                        self.logger.info("work:------------------------------未获取到云台位置,3秒后重试")
                        time.sleep(3)
                        camPosStr = self.getCameraPositon()
                    # self.updatePTZlineEdtSig.emit(camPosStr)
                    self.num += 1
                    #发射更新进度条信号
                    # self.setPrgBarValSig.emit(self.num)
                else:
                    self.logger.info("work:setPosition 失败！")
                    self.logger.info("work:self.num--%s"%self.num)
                    self.startWork("retry")
                    break #跳出循环
        elif flag == "retry":
            for i in range(self.num, len(self.positionList)):
                self.logger.info("\n\n————————————————————————————————  %d  ————————————————————————————————"%int(self.num))
                self.logger.info("work:retry-国悦府-第%d张图片"%i)
                self.logger.info("work:retry-国悦府-共%d张图片"%(len(self.positionList)))
                self.num = i
                # 设定位置
                self.logger.info("work:1 设置云台位置")
                setPositionRes = self.setPositonRequest(self.positionList[self.num]) 
                if setPositionRes != None:
                    self.logger.info("work:retry-setPosition 成功！")
                    #延时3秒
                    time.sleep(3)
                    #截图
                    self.logger.info("work:retry-2 截图")
                    imageBase64Str = self.cptureRequests()
                    while imageBase64Str == None:
                        self.logger.info("work:retry-------------------------------未获取到图片,3秒后重试")
                        time.sleep(3)
                        imageBase64Str = self.cptureRequests()
                    #转成图片
                    imgdata = self.base64ToImage(imageBase64Str)
                    #保存
                    saveImagePath = self.saveCapImg(imgdata)
                    #发送展示抓取图像信号
                    # self.getImageSig.emit(saveImagePath)
                    #获取相机位置
                    self.logger.info("work:retry-3 获取云台位置")
                    camPosStr = self.getCameraPositon()
                    while camPosStr == None:
                        self.logger.info("work:retry-------------------------------未获取到云台位置,3秒后重试")
                        time.sleep(3)
                        camPosStr = self.getCameraPositon()
                    # self.updatePTZlineEdtSig.emit(camPosStr)
                    self.num += 1
                    #发射更新进度条信号
                    # self.setPrgBarValSig.emit(self.num)
                else:
                    self.logger.info("work:retry-setPosition 失败！")
                    self.logger.info("work:retry-self.num--%d" %(self.num))
                    self.startWork("retry")
                    break #跳出循环
        self.logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!图片抓取完毕\n")
        self.num = 0
        #发射抓图完毕信号
        self.captureImgFinishSig.emit()

    #获取token
    def getTokenRequest(self):
        #设置超时时间 10s
        socket.setdefaulttimeout(10)
        #header 信息
        headers = {'Authorization': 'bm9uZ2RhMTpuZDEyMzQ1Ng=='}
        #get请求
        tokenUrl = "http://218.201.180.118:9840/apiserver/v1/user/authentication-token"
        try:
            tokenResponse = requests.get(tokenUrl, headers=headers,timeout=(3,7))
        except:
            self.logger.info("------------------------------token获取失败")
            self.logger.exception(sys.exc_info())
            tokenResponse = None
        if tokenResponse == None:
           return None
        #如果token不为空
        else:
            #从返回值中获取token，保存到全局变量token中
            data = json.loads(tokenResponse.text)
            token = data.get("data")
            if token == None :
                self.logger.info("Error 获取token失败！")
                return None
            else:
                self.logger.info("获取Token成功！" + ":" + token)
                return token
        
    #设置云台位置
    def setPositonRequest(self, positionStr):
        #获取token
        token = self.getTokenRequest()
        if token != None:
            #取出水平、俯仰、放大值
            tmpList = positionStr.split("-")
            panStr = tmpList[0]
            tiltStr = tmpList[1]
            zoomStr = tmpList[2]
            self.logger.info("设置云台位置:%s-%s-%s"%(panStr,tiltStr,zoomStr))
            #组成json
            postData = dict(camera_id="37090200001311000980", pan = int(panStr), tilt=int(tiltStr), zoom=int(zoomStr))
            setPostionUrl = "http://218.201.180.118:9840/apiserver/v1//device/ptz/postion?token="+token
            try:
                setPostionResponds = requests.post(setPostionUrl, json=postData,timeout=(3,7))
            except:
                self.logger.info('设置云台位置----------------------------------设置云台位置失败')
                self.logger.exception(sys.exc_info())
                setPostionResponds = None
            if setPostionResponds != None:
                self.logger.info("设置云台位置:成功:%s"%setPostionResponds.text)
                data = json.loads(setPostionResponds.text)
                setPositionRes = data.get("data")
                return setPositionRes 
            else:
                self.logger.info('设置云台位置----------------------------------设置云台位置失败')
                return None
        else:
            self.logger.info('设置云台位置------------------------------设置云台位置时，获取token失败')
            return None
        
    #获取图像
    def cptureRequests(self):
        #设置超时时间 10s
        socket.setdefaulttimeout(10)
        token = self.getTokenRequest()# 获取token
        if token != None:
            frontStr = "http://218.201.180.118:9840/apiserver/v1//device/video/capture?token="
            backStr = "&camera_id=37090200001311000980&stream_type=0&is_local=1&data_mode=1&pic_num=1"
            cptureUrl = frontStr + token + backStr
            try:
                cptureResponse = requests.get(cptureUrl,timeout=(3,7))
            except:
                self.logger.info("获取图像:------------------------------获取图像失败")
                self.logger.exception(sys.exc_info())
                cptureResponse = None
            if cptureResponse != None:
                tmpStr1 = json.loads(cptureResponse.text)
                dataDic = tmpStr1.get("data")#取出data
                if dataDic == None:
                    self.logger.info("获取图像:dataDic_______为空:%s"%dataDic)
                    return None
                else:
                    self.logger.info("获取图像:成功:dataDic_______不为空")
                    picBufDic = dataDic[0].get("pic_buf")#取出data中的pic_buf
                    tmpStr2 = picBufDic.split(',')
                    imgStr= tmpStr2[-1]
                    return imgStr # 返回base64码
            else:
                self.logger.info('获取图像:------------------------------获取图象失败')
                return None
        else:
            self.logger.info('获取图像:------------------------------获取图象过程中，获取token失败')
            return None

    #获取云台位置
    def getCameraPositon(self):
        # 设置超时时间 10s
        socket.setdefaulttimeout(10)
        # 获取token
        token = self.getTokenRequest()
        if token != None:
            frontStr = "http://218.201.180.118:9840/apiserver/v1/device/ptz/postion?token="
            backStr = "&camera_id=37090200001311000980"
            cptureUrl = frontStr + token + backStr
            try:
                cptureResponse = requests.get(cptureUrl,timeout=(3,7))
            except:
                self.logger.info("获取云台位置:-------------------------------获取云台位置失败")
                self.logger.exception(sys.exc_info())
                cptureResponse = None
            if cptureResponse != None:
                tmpStr1 = json.loads(cptureResponse.text)
                dataDic = tmpStr1.get("data")#取出data
                if dataDic == None:
                    self.logger.info("获取云台位置:data信息为空:dataDic:%s"%dataDic)
                    tmpStr = self.positionList[self.num]
                    #取出水平、俯仰、放大值
                    tmpList = tmpStr.split("-")
                    panStr = tmpList[0]
                    tiltStr = tmpList[1]
                    zoomStr = tmpList[2]
                    camPosiStr = panStr+"-"+tiltStr+"-"+zoomStr
                    return None
                else:
                    pan = str(dataDic["pan"])#取出data中的pic_buf
                    tilt = str(dataDic["tilt"])#取出data中的pic_buf
                    zoom = str(dataDic["zoom"])#取出data中的pic_buf
                    camPosiStr = pan+"-"+tilt+"-"+zoom
                    self.logger.info("获取云台位置:成功:camPosiStr:%s"%camPosiStr)
                    return camPosiStr
            else:
                self.logger.info("获取云台位置:------------------------------获取当前云台位置失败")
                return None
        else:
            self.logger.info('获取云台位置:------------------------------获取当前云台位置过程中，获取token失败')
            return None    
        
    #base64转图片
    def base64ToImage(self,base64Str):
        imgdata = base64.b64decode(base64Str)
        return imgdata

    # 保存抓取到的图片
    def saveCapImg(self, imgdata):
        if self.num < len(self.positionList):
            panStr = self.positionList[self.num].split('-')[0]
            tiltStr = self.positionList[self.num].split('-')[1]
            zoomStr = self.positionList[self.num].split('-')[2]
            self.curImgSavePath = self.capSavePath + panStr +"-" + tiltStr + "-" + zoomStr +".jpg"
        file = open(self.curImgSavePath, 'wb')
        file.write(imgdata)
        file.close()

        # return self.curImgSavePath



if __name__ == '__main__':
    app = QApplication(sys.argv)
    capturePicWidget = CapturePicWidget()
    token = capturePicWidget.getTokenRequest()
    print(token)