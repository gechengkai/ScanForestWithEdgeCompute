from PyQt5.QtWidgets import QApplication, QWidget
import glob, os, socket, sys
from pathlib import Path
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

CAP_SAVE_PATH = 'EdgeTerminal\\detectDir\\taian\\capImg\\'
DETECT_RES_PATH = 'EdgeTerminal\\detectDir\\taian\\detect\\'

class PassbackWidget(QWidget):
    # 信号

    def getPicPaths(self):
        picPaths = []
        txtPaths = glob.glob(os.path.join(DETECT_RES_PATH, '*.txt')) #获取图片路径
        print("\ntxtPahts: ",txtPaths)
        for txtPath in txtPaths:
            txtName = Path(txtPath).stem
            picPath = CAP_SAVE_PATH + txtName + '.jpg'
            picPaths.append(picPath)

        return picPaths

    def passback(self, picPaths):

        HOST,PORT = "192.168.31.53",5150
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
            # print(f'Picture baseName: {picName}')
            picName = picName.ljust(64, "@") #补足64位
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
    

    def main(self):
        picPaths = self.getPicPaths()
        print("\nPicPaths: ",picPaths)
        self.passback(picPaths)
        # 回传完毕后清空文件夹
        # shutil.rmtree(DETECT_RES_PATH) #清空文件夹
        # shutil.rmtree(CAP_SAVE_PATH) #清空文件夹


if __name__ == '__main__':
    app = QApplication(sys.argv)
    passbackWidget = PassbackWidget()
    passbackWidget.main()
    # sys.exit(app.exec_())