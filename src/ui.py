# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/PCR.Benchmark.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class benchmarkUI(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1672, 912)
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(380, 50, 1271, 731))
        self.widget.setObjectName("widget")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 121, 16))
        self.label_2.setObjectName("label_2")
        self.gtR = QtWidgets.QTextBrowser(self.widget)
        self.gtR.setGeometry(QtCore.QRect(1030, 40, 191, 111))
        self.gtR.setObjectName("gtR")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(1030, 20, 141, 16))
        self.label_4.setObjectName("label_4")
        self.gtT = QtWidgets.QTextBrowser(self.widget)
        self.gtT.setGeometry(QtCore.QRect(1030, 180, 191, 21))
        self.gtT.setObjectName("gtT")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setGeometry(QtCore.QRect(1030, 160, 151, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setGeometry(QtCore.QRect(1030, 230, 141, 16))
        self.label_6.setObjectName("label_6")
        self.estR = QtWidgets.QTextBrowser(self.widget)
        self.estR.setGeometry(QtCore.QRect(1030, 250, 191, 111))
        self.estR.setObjectName("estR")
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setGeometry(QtCore.QRect(1030, 370, 151, 16))
        self.label_7.setObjectName("label_7")
        self.estT = QtWidgets.QTextBrowser(self.widget)
        self.estT.setGeometry(QtCore.QRect(1030, 390, 191, 21))
        self.estT.setObjectName("estT")
        self.label_8 = QtWidgets.QLabel(self.widget)
        self.label_8.setGeometry(QtCore.QRect(1030, 430, 141, 16))
        self.label_8.setObjectName("label_8")
        self.label_12 = QtWidgets.QLabel(self.widget)
        self.label_12.setGeometry(QtCore.QRect(1030, 470, 41, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.widget)
        self.label_13.setGeometry(QtCore.QRect(1030, 450, 41, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.widget)
        self.label_14.setGeometry(QtCore.QRect(1030, 490, 91, 16))
        self.label_14.setObjectName("label_14")
        self.rMAE = QtWidgets.QTextBrowser(self.widget)
        self.rMAE.setGeometry(QtCore.QRect(1070, 470, 161, 20))
        self.rMAE.setObjectName("rMAE")
        self.rMSE = QtWidgets.QTextBrowser(self.widget)
        self.rMSE.setGeometry(QtCore.QRect(1070, 450, 161, 20))
        self.rMSE.setObjectName("rMSE")
        self.rIsotropic = QtWidgets.QTextBrowser(self.widget)
        self.rIsotropic.setGeometry(QtCore.QRect(1130, 490, 101, 20))
        self.rIsotropic.setObjectName("rIsotropic")
        self.label_15 = QtWidgets.QLabel(self.widget)
        self.label_15.setGeometry(QtCore.QRect(1030, 570, 31, 16))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.widget)
        self.label_16.setGeometry(QtCore.QRect(1030, 550, 41, 16))
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.widget)
        self.label_17.setGeometry(QtCore.QRect(1030, 590, 91, 16))
        self.label_17.setObjectName("label_17")
        self.tMAE = QtWidgets.QTextBrowser(self.widget)
        self.tMAE.setGeometry(QtCore.QRect(1070, 570, 161, 20))
        self.tMAE.setObjectName("tMAE")
        self.tMSE = QtWidgets.QTextBrowser(self.widget)
        self.tMSE.setGeometry(QtCore.QRect(1070, 550, 161, 20))
        self.tMSE.setObjectName("tMSE")
        self.tIsotropic = QtWidgets.QTextBrowser(self.widget)
        self.tIsotropic.setGeometry(QtCore.QRect(1130, 590, 101, 20))
        self.tIsotropic.setObjectName("tIsotropic")
        self.label_18 = QtWidgets.QLabel(self.widget)
        self.label_18.setGeometry(QtCore.QRect(1030, 530, 141, 16))
        self.label_18.setObjectName("label_18")
        self.label_11 = QtWidgets.QLabel(self.widget)
        self.label_11.setGeometry(QtCore.QRect(20, 680, 161, 16))
        self.label_11.setObjectName("label_11")
        self.label_19 = QtWidgets.QLabel(self.widget)
        self.label_19.setGeometry(QtCore.QRect(210, 680, 161, 16))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.widget)
        self.label_20.setGeometry(QtCore.QRect(390, 680, 171, 16))
        self.label_20.setObjectName("label_20")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.widget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 40, 971, 631))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.PCViewer = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.PCViewer.setContentsMargins(0, 0, 0, 0)
        self.PCViewer.setObjectName("PCViewer")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setGeometry(QtCore.QRect(380, 810, 1271, 71))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget = QtWidgets.QWidget(self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 20, 1091, 32))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.registrationButton = QtWidgets.QPushButton(self.layoutWidget)
        self.registrationButton.setObjectName("registrationButton")
        self.horizontalLayout_2.addWidget(self.registrationButton)
        self.nextButton = QtWidgets.QPushButton(self.layoutWidget)
        self.nextButton.setObjectName("nextButton")
        self.horizontalLayout_2.addWidget(self.nextButton)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.progressBar = QtWidgets.QProgressBar(self.layoutWidget)
        self.progressBar.setEnabled(True)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_2.addWidget(self.progressBar)
        self.processBarLabel = QtWidgets.QLabel(self.layoutWidget)
        self.processBarLabel.setObjectName("processBarLabel")
        self.horizontalLayout_2.addWidget(self.processBarLabel)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(20, 60, 351, 411))
        self.groupBox.setObjectName("groupBox")
        self.layoutWidget1 = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 30, 301, 26))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget1)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.comboBox = QtWidgets.QComboBox(self.layoutWidget1)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout.addWidget(self.comboBox)
        self.layoutWidget2 = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 70, 301, 251))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_10.setObjectName("label_10")
        self.verticalLayout.addWidget(self.label_10)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_3.addWidget(self.lineEdit)
        self.toolButton = QtWidgets.QToolButton(self.layoutWidget2)
        self.toolButton.setObjectName("toolButton")
        self.horizontalLayout_3.addWidget(self.toolButton)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.listView = QtWidgets.QListView(self.layoutWidget2)
        self.listView.setObjectName("listView")
        self.verticalLayout.addWidget(self.listView)
        self.loadButton = QtWidgets.QPushButton(self.groupBox)
        self.loadButton.setGeometry(QtCore.QRect(190, 330, 113, 32))
        self.loadButton.setObjectName("loadButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "PCR.Benchmark"))
        self.label_2.setText(_translate("Form", "Point Clouds View:"))
        self.label_4.setText(_translate("Form", "Ground truth rotation:"))
        self.label_5.setText(_translate("Form", "Ground truth translation:"))
        self.label_6.setText(_translate("Form", "Estimation rotation:"))
        self.label_7.setText(_translate("Form", "Estimation translation:"))
        self.label_8.setText(_translate("Form", "Rotation errors"))
        self.label_12.setText(_translate("Form", "MAE:"))
        self.label_13.setText(_translate("Form", "MSE:"))
        self.label_14.setText(_translate("Form", "Isotropic error:"))
        self.label_15.setText(_translate("Form", "MAE:"))
        self.label_16.setText(_translate("Form", "MSE:"))
        self.label_17.setText(_translate("Form", "Isotropic error:"))
        self.label_18.setText(_translate("Form", "Translation errors"))
        self.label_11.setText(_translate("Form", "Green: Source Point Cloud"))
        self.label_19.setText(_translate("Form", "Red: Target Point Cloud"))
        self.label_20.setText(_translate("Form", "Blue: Estimated Point Cloud"))
        self.registrationButton.setText(_translate("Form", "Registration"))
        self.nextButton.setText(_translate("Form", "Next"))
        self.label_3.setText(_translate("Form", "PCR Process:"))
        self.processBarLabel.setText(_translate("Form", "-- / --"))
        self.groupBox.setTitle(_translate("Form", "PCR Settings"))
        self.label.setText(_translate("Form", "PCR Method:"))
        self.label_10.setText(_translate("Form", "Dataset Files(HDF5) Directory:"))
        self.toolButton.setText(_translate("Form", "..."))
        self.loadButton.setText(_translate("Form", "Load"))
