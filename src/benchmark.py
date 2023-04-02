import sys
import os
sys.path.append('.')
import numpy as np
from baselines.APIs import DGMNetAPI, DGMAPI, ICPAPI
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import pyqtgraph.opengl as gl
from src.ui import benchmarkUI
from process import PCRProcess

PCR_VIEW_EXPAND=3
PCR_VIEW_POINTSIZE=0.08

class MainWindow(QMainWindow,benchmarkUI):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # set point cloud viewer
        self.PCWidget=gl.GLViewWidget()
        self.PCWidget.setBackgroundColor(96,96,96,255)
        self.PCViewer.addWidget(self.PCWidget)
        self.progressBar.setValue(0)

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
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(self.process.getTotal())
        self.move()
        
    def registration(self):
        self.verifyPCRAPI()
        est_R,est_t,time_cost = self.process.registration(self.PCRAPI, self.getPCRMetohd())
        est_pc=self.process.transform(est_R, est_t)
        est_plot=gl.GLScatterPlotItem()
        est_plot.setData(pos=PCR_VIEW_EXPAND*est_pc, color=(0., 0., 1., 1), size=PCR_VIEW_POINTSIZE, pxMode=False)
        self.PCWidget.addItem(est_plot)
        R_MSE,R_MAE,R_isotropic,t_MSE,t_MAE,t_isotropic=self.process.metrics(est_R, est_t)
        est_R=np.around(est_R,4)
        est_t=np.around(est_t,4)
        self.estR.setText(str(est_R))
        self.estT.setText(str(est_t))
        self.rMSE.setText(str(R_MSE))
        self.rMAE.setText(str(R_MAE))
        self.rIsotropic.setText(str(R_isotropic))
        self.tMSE.setText(str(t_MSE))
        self.tMAE.setText(str(t_MAE))
        self.tIsotropic.setText(str(t_isotropic))

    def move(self):
        src_pc,tgt_pc,gt_R,gt_t,cur,total=self.process.move()
        self.progressBar.setValue(self.process.getCurrent())
        gt_R=np.around(gt_R,4)
        gt_t=np.around(gt_t,4)
        self.gtR.setText(str(gt_R))
        self.gtT.setText(str(gt_t))
        self.processBarLabel.setText(str(cur)+' / '+str(total))
        # view point cloud
        src_plot=gl.GLScatterPlotItem()
        src_plot.setData(pos=PCR_VIEW_EXPAND*src_pc, color=(0., 1., 0., 1), size=PCR_VIEW_POINTSIZE, pxMode=False)
        tgt_plot=gl.GLScatterPlotItem()
        tgt_plot.setData(pos=PCR_VIEW_EXPAND*tgt_pc, color=(1., 0., 0., 1), size=PCR_VIEW_POINTSIZE, pxMode=False)
        
        self.PCWidget.clear()
        self.PCWidget.addItem(src_plot)
        self.PCWidget.addItem(tgt_plot)

if __name__ == '__main__':
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec_())