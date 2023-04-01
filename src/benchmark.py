import sys
import os
sys.path.append('.')
from baselines.APIs import DGMNetAPI, DGMAPI, ICPAPI
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from src.ui import benchmarkUI
from process import PCRProcess

class MainWindow(QMainWindow,benchmarkUI):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # verify PCR Methods
        self.comboBox.addItem('DGM-Net')
        self.comboBox.addItem('DGM')
        self.comboBox.addItem('ICP')
        self.comboBox.currentTextChanged.connect(self.verifyPCRAPI)
        
        # verify Datasets HDF5 Files directory
        self.listModel=QStandardItemModel()
        self.toolButton.clicked.connect(self.verifyDatasetDir)

        # load
        self.loadButton.clicked.connect(self.load)
        
        # move
        self.nextButton.clicked.connect(self.move)

        # registration
        self.registrationButton.clicked.connect(self.registration)

    def getPCRMetohd(self):
        return self.comboBox.currentText()

    def verifyDatasetDir(self):
        self.datasetDir=QFileDialog.getExistingDirectory(caption="select directory contains point cloud data files(.h5)",directory="/")
        self.lineEdit.setText(self.datasetDir)
        fs=os.listdir(self.datasetDir)
        for f in fs:
            if f.endswith('.h5') or f.endswith('hdf5'):
                item=QStandardItem(f)
                item.setCheckable(True)
                self.listModel.appendRow(item)
        self.listView.setModel(self.listModel)

    def getTestFiles(self):
        res=[]
        for index in range(self.listModel.rowCount()):
            if(self.listModel.item(index).checkState()):
                res.append(self.datasetDir+'/'+self.listModel.item(index).text())
        return res

    def verifyPCRAPI(self):
        methodName=self.getPCRMetohd()
        if methodName=='DGM-Net':
            self.PCRAPI=DGMNetAPI
        elif methodName=='DGM':
            self.PCRAPI=DGMAPI
        elif methodName=='ICP':
            self.PCRAPI=ICPAPI

    def load(self):
        self.process=PCRProcess(self.getTestFiles())
        
    def registration(self):
        self.process.registration(self.PCRAPI)
        pass
        
    def move(self):
        pass

if __name__ == '__main__':
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec_())