# -*- coding: utf-8 -*-
"""
GUI to process Memristor experiments

By Cons, 2017
"""

import sys, random, csv, pdb, re
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from test_design import Ui_MainWindow

from matplotlib.backends.backend_qt5agg import \
  FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
  NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# class IntervalTypeList(QtWidgets.QWidget):
#   def __init__(self, parent=None):
#     QtWidgets.QWidget.__init__(self, parent=parent)
#     lay = QtWidgets.QVBoxLayout(self)
#     for i in range(4):
#       lay.addWidget(QtWidgets.QPushButton("{}".format(i)))


class MyProgram(Ui_MainWindow):
  def __init__(self, dialog):
    super(MyProgram, self).__init__()
    self.setupUi(dialog)
    self.m = MyMplCanvas(self.dockWidgetContents)
    self.mpl_toolbar = NavToolbar(self.m, self.dockWidgetContents)

    self.dataConverted = []
    self.intervals2plot_eval = set()
    self.intervals2plot_train = set()

# ==============================================================================
#     Connect signals
# ==============================================================================

    self.openButn.clicked.connect(lambda: self.openFile(dialog))

    self.drawButn.clicked.connect(lambda:
          self.drawPlotTop(dialog, np.arange(0,(self.rawRangeSld.value())*5,5),
          self.dataConverted[
              self.rawStartSld.value() - 1 :
              self.rawStartSld.value() + self.rawRangeSld.value() - 1]) )

    self.rawStartSld.valueChanged.connect(lambda:
          self.updateSpinBox(self.startInp, self.rawStartSld.value()) )

    self.rawRangeSld.valueChanged.connect(lambda:
          self.updateSpinBox(self.rangeInp, self.rawRangeSld.value()) )

    self.integrateButn.clicked.connect(lambda:
          self.integrateData(self.dataConverted))

    self.integrateButn_2.clicked.connect(lambda:
          self.integrateDataCustomIntervals(self.dataConverted))

    self.startInp.valueChanged.connect(lambda:
          self.rawStartSld.setValue(self.startInp.value()))

    self.rangeInp.valueChanged.connect(lambda:
          self.rawRangeSld.setValue(self.rangeInp.value()))
    self.rangeInp.valueChanged.connect(lambda:
          self.changeRangeAdjustStart())

    self.recalButn.clicked.connect(self.convertData)

    self.addIntervalButton.clicked.connect(self.addInterval)
    self.addIntervalButton_2.clicked.connect(self.addInterval_2)

    self.removeIntervalButton.clicked.connect(self.removeInterval)
    self.removeIntervalButton_2.clicked.connect(self.removeInterval_2)

    self.intervalTable.itemClicked.connect(self.handleIntervalTable1Click)
    self.intervalTable_2.itemClicked.connect(self.handleIntervalTable2Click)

    self.reopenButn.clicked.connect(self.reopenFile)
# ==============================================================================
#       End of connections
# ==============================================================================

  def openFile(self, dialog):
    self.fname, _ = QtWidgets.QFileDialog.getOpenFileName \
              (dialog, "QFileDialog.getOpenFileName()",
                "/media/cons/DATA-1.2TB1/Memristors/Arduino experiments/PulsesList",
               "All Files (*);;Logs (*.log)")
    if self.fname:
      self.dataRaw = self.openFileReader(self.fname)
      self.maxRange = min(100000, len(self.dataRaw[1]))
      self.convertData()
      self.updateControlsOnFileopen()
      self.filenameLabel.setText('The File is open')
      self.statusbar.showMessage(self.fname)

  def reopenFile(self):
    if self.fname:
      self.dataRaw = self.openFileReader(self.fname)
      self.maxRange = min(100000, len(self.dataRaw[1]))
      self.convertData()
      self.updateControlsOnFileopen()
      self.filenameLabel.setText('The File is open')
      self.statusbar.showMessage(self.fname)

  def handleIntervalTable1Click(self, item):
    if item.checkState() == QtCore.Qt.Checked:
      self.intervals2plot_train.add(item.row())
      self.filenameLabel.setText(str(self.intervals2plot_train))
    if item.checkState() == QtCore.Qt.Unchecked:
      try:
        self.intervals2plot_train.remove(item.row())
        self.filenameLabel.setText(str(self.intervals2plot_train))
      except:
        pass

  def handleIntervalTable2Click(self, item):
    if item.checkState() == QtCore.Qt.Checked:
      self.intervals2plot_eval.add(item.row())
      self.filenameLabel.setText(str(self.intervals2plot_eval))
    if item.checkState() == QtCore.Qt.Unchecked:
      try:
        self.intervals2plot_eval.remove(item.row())
        self.filenameLabel.setText(str(self.intervals2plot_eval))
      except:
        pass

  def addInterval(self):
    currentRow = self.intervalTable.rowCount()
    self.intervalTable.insertRow(currentRow)
    self.intervalTable.setItem(currentRow, 0, QtWidgets.QTableWidgetItem("20"))
    checkBoxItem = QtWidgets.QTableWidgetItem()
    checkBoxItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
    checkBoxItem.setCheckState(QtCore.Qt.Unchecked)
    self.intervalTable.setItem(currentRow, 1, checkBoxItem)

  def addInterval_2(self):
    currentRow = self.intervalTable_2.rowCount()
    self.intervalTable_2.insertRow(currentRow)
    self.intervalTable_2.setItem(currentRow, 0, QtWidgets.QTableWidgetItem("20"))
    checkBoxItem = QtWidgets.QTableWidgetItem()
    checkBoxItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
    checkBoxItem.setCheckState(QtCore.Qt.Unchecked)
    self.intervalTable_2.setItem(currentRow, 1, checkBoxItem)

  def removeInterval(self):
    currentRow = self.intervalTable.rowCount()
    self.intervalTable.removeRow(currentRow-1)

  def removeInterval_2(self):
    currentRow = self.intervalTable_2.rowCount()
    self.intervalTable_2.removeRow(currentRow-1)

  def convertData(self):
    self.dataConverted = [float((float(x) -
                                 float(self.inputZeroI.text())) * 4.883 *
                                  float(self.inputYS.text()) /
                                  float(self.inputR.text()))
                          for x in self.dataRaw[3]]

  def drawPlotTop(self, dialog, dataX, dataY, plot_style='-', hold=0):
    if (self.holdPlotChk.checkState() == 0) & (hold!=1):
      self.m.axes1.clear()
    self.m.plotTop(dataX, dataY, plot_style)

  def drawPlotBottom(self, dialog, data, plot_style='-', hold=0, colorB='b'):
    if (self.holdPlotChk.checkState() == 0) & (hold!=1):
      self.m.axes2.clear()
    self.m.plotBottom(data, plot_style, colorB)

  def openFileReader(self, fname):
    column_list1 = []
    column_list2 = []
    column_list3 = []
    column_list4 = []
    iRow = 0

    with open(fname, 'r', encoding='UTF8') as data_file:
      reader = csv.reader(data_file)

      for dataRow in reader:
        try:
          col1, col2, col3, col4 = dataRow[0].split()
          column_list1.append(col1)
          column_list2.append(col2)
          column_list3.append(col3)
          column_list4.append(col4)
          iRow += 1
        except:
          print(iRow)

    return column_list1, column_list2, column_list3, column_list4

  def updateControlsOnFileopen(self):
    self.updateSlider(self.rawStartSld, len(self.dataRaw[1]), 1,
                      minIn=1)
    self.updateSlider(self.rawRangeSld, self.maxRange, 10000, 100, 100)
    self.updateLabel(self.rawRangeLL, self.rawRangeSld.minimum())
    self.updateLabel(self.rawStartLL, 1)
    self.updateLabel(self.rawStartRL, len(self.dataRaw[1]))
    self.updateLabel(self.rawRangeRL, self.maxRange)
    self.startInp.setMaximum(len(self.dataRaw[1]))
    self.rangeInp.setMaximum(self.maxRange)

  def updateSlider(self, slider, maxIn, value_to_set=100,
                   tick_step=100, minIn=0):
    slider.setMinimum(minIn)
    slider.setMaximum(maxIn)
    slider.setValue(value_to_set)
    slider.setTickInterval(maxIn//tick_step*100)
    slider.setSingleStep(tick_step)
    slider.setPageStep(tick_step*50)

  def changeRangeAdjustStart(self):
    self.startInp.setMaximum(max(len(self.dataRaw[1]) - self.rangeInp.value(),
                                 1))
    self.rawStartSld.setMaximum(self.startInp.maximum())
    self.updateLabel(self.rawStartRL, self.startInp.maximum())

  def updateSpinBox(self, spinBox, dataIn):
#    spinBox.blockSignals(True)
    spinBox.setValue(int(dataIn))
#    spinBox.blockSignals(False)

  def updateLabel(self, label, dataIn):
    label.setText(str(dataIn))

  def integrateData(self, dataIn):
    # try:
      self.integrated_data = list()
      p1 = int(self.inp_pulse1.text())
      p1p = int(self.inp_pulse1_prior.text())
      p1r = int(self.inp_pulse1_rest.text())
      p2 = int(self.inp_pulse2.text())
      p2r = int(self.inp_pulse2_rest.text())
      one_period = p1p + p1 + p1r + p2 + p2r

      for i in range(0, len(dataIn)//one_period):
        period_data = dataIn[one_period*i:one_period*(i+1)]
        period_data = [float(xAux) for xAux in period_data]
        self.integrated_data.append([sum(period_data[p1p:p1p + p1]),
                                     sum(period_data[p1p + p1 + p1r:
                                     p1p + p1 + p1r + p2])])
      color1 = ("#%06x" % random.randint(0, 0xFFFFFF))
      self.drawPlotBottom(dialog, [x[0] for x in
                         self.integrated_data], '.', colorB=color1)
      self.drawPlotBottom(dialog, [x[1] for x in
                         self.integrated_data], '.', 1, colorB=color1)
    # except:
    #   self.filenameLabel.setText('GAVNO')

  def integrateDataCustomIntervals(self, dataIn):
    # try:
    number_of_intervals_train = self.intervalTable.rowCount()
    number_of_intervals_eval = self.intervalTable_2.rowCount()
    intervals_train = []
    intervals_eval = []
    for interval in range(0, number_of_intervals_train):
      intervals_train.append(int(self.intervalTable.item(interval, 0).text()))

    for interval in range(0, number_of_intervals_eval):
      intervals_eval.append(int(self.intervalTable_2.item(interval, 0).text()))

    self.filenameLabel.setText(str(intervals_train))

    self.integrated_data = list()
    # p1 = int(self.inp_pulse1.text())
    # p1p = int(self.inp_pulse1_prior.text())
    # p1r = int(self.inp_pulse1_rest.text())
    # p2 = int(self.inp_pulse2.text())
    # p2r = int(self.inp_pulse2_rest.text())
    # one_period = p1p + p1 + p1r + p2 + p2r

    self.training_data = list()
    self.evaluation_data = list()
    self.dataSorted = list()
    self.training_cycles_number = int(self.numberTrainingCyclesLine.text())
    self.eval_cycles_number = int(self.numberEvalCyclesLine.text())
    self.training_intervals_number = len(intervals_train)
    self.eval_intervals_number = len(intervals_eval)
    self.training_datapoints_number = sum(intervals_train)
    self.eval_datapoints_number = sum(intervals_eval)

    dictKeys = ['train', 'eval']

    one_training_cycle_length = 0
    for v in intervals_train:
      one_training_cycle_length += v

    one_eval_cycle_length = 0
    for v in intervals_eval:
      one_eval_cycle_length += v

    one_epoch_length = one_training_cycle_length*self.training_cycles_number \
                       + one_eval_cycle_length*self.eval_cycles_number

    """
    FORMAT OF DATA: dataSorted[epoch number]['train'|'eval'][interval number][corresponding points]
    """

    try:
      for i in range(0, len(dataIn)//one_epoch_length):
        epoch_data = dataIn[one_epoch_length*i:one_epoch_length*(i+1)]
        epoch_data_iter = iter([float(xAux) for xAux in epoch_data])

        intra_training = []
        for iCycle in range(self.training_cycles_number):
          intra_cycle = []
          for j in range(self.training_intervals_number):
            intra_interval = []
            for k in range(intervals_train[j]):
              intra_interval.append(next(epoch_data_iter))
            intra_cycle.append(intra_interval)
          intra_training.append(intra_cycle)

        intra_eval = []
        for iCycle in range(self.eval_cycles_number):
          intra_cycle = []
          for j in range(self.eval_intervals_number):
            intra_interval = []
            for k in range(intervals_eval[j]):
              intra_interval.append(next(epoch_data_iter))
            intra_cycle.append(intra_interval)
          intra_eval.append(intra_cycle)

        self.dataSorted.append(dict(train=intra_training, eval=intra_eval))

    except:
      self.filenameLabel.setText('Intervaly GAVNO')

    color1 = ("#%06x" % random.randint(0, 0xFFFFFF))

    for data in self.dataSorted:
      for cycles in data['train']:
        for interval2plot in self.intervals2plot_train:
          self.integrated_data.append(sum(cycles[interval2plot]))

      for cycles in data['eval']:
        for interval2plot in self.intervals2plot_eval:
          self.integrated_data.append(sum(cycles[interval2plot]))

    self.drawPlotBottom(dialog, [x for x in
                                 self.integrated_data], '.', colorB=color1)

    self.trainCycLengthLabel.setText(str(one_training_cycle_length))
    self.evalCycLengthLabel.setText(str(one_eval_cycle_length))
    self.epochLengthLabel.setText(str(one_epoch_length))

class MyMplCanvas(FigureCanvas):
  def __init__(self, parent=None, widthIn=15, heightIn=12):
    fig = Figure(figsize=(widthIn, heightIn))
    self.axes1 = fig.add_subplot(2,1,1)
    self.axes1.set_title('Pulses [mA]')
    self.axes2 = fig.add_subplot(2,1,2)
    self.axes2.set_title('Integrated pulses [uC]')

    FigureCanvas.__init__(self, fig)
    self.setParent(parent)
    self.lines = self.axes1.plot(0);

    FigureCanvas.setSizePolicy(self,
                   QtWidgets.QSizePolicy.Expanding,
                   QtWidgets.QSizePolicy.Expanding)
    FigureCanvas.updateGeometry(self)

  def plotTop(self, x, y, plot_style='-'):
    self.lines.append(self.axes1.plot(x, y, plot_style))
    self.axes1.set_title('Pulses [mA]')
    # to add: removing arbitrary lines
#==============================================================================
#     try:
#       print('Before removal')
#       print(self.axes1.lines)
#       print(self.lines)
#       self.lines.remove(self.lines[2])
#       print('After removal')
#       print(self.axes1.lines)
#       print(self.lines)
#     except:
#       exceptName = sys.exc_info()[0]
#       print(exceptName)
#==============================================================================
    self.draw()

  def plotBottom(self, data, plot_style='-', colorB='b'):
    self.lines.append(self.axes2.plot(data, plot_style, color=colorB))
    self.axes2.set_title('Integrated pulses [uC]')
    self.draw()

  def plotPulse(self, length):
    self.axes1.plot([length]*100)
    self.draw()

  def getData(self):
    return self.data


class DynamicCanvas(MyMplCanvas):
  """A canvas that updates itself every second with a new plot."""

  def __init__(self, *args, **kwargs):
    MyMplCanvas.__init__(self, *args, **kwargs)
    timer = QtCore.QTimer(self)
    timer.timeout.connect(self.update_figure)
    timer.start(1000)

  def compute_initial_figure(self):
    self.axes1.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

  def update_figure(self):
    # Build a list of 4 random integers between 0 and 10 (both inclusive)
    l = [random.randint(0, 10) for i in range(4)]

    self.axes1.plot([0, 1, 2, 3], l, 'r')
    self.draw()

#%% Main
if __name__ == '__main__':
  app = QtWidgets.QApplication(sys.argv)
  dialog = QtWidgets.QMainWindow()

  prog = MyProgram(dialog)
  dialog.show()
  sys.exit(app.exec_())


#class PlotCanvas(FigureCanvas):
#  def __init__(self, parent=None, width=5, height=4):
#    fig = Figure()
#    self.axes = fig.add_subplot(111)
#    FigureCanvas.__init__(self, fig)
#    self.setParent(parent)
##    FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
##    FigureCanvas.updateGeometry(self)
#    self.plotTop()
#
#  def plotTop(self):
#    self.data = [random.random() for i in range(25)]
#    ax = self.figure.add_subplot(111)
#    ax.plot(self.data, 'r-')
#    ax.set_title('PyQt Matplotlib Example')
#    self.draw()
#
#  def plotPulse(self, length):
#    ax = self.figure.add_subplot(111)
#    ax.plot([length]*100)
#
#  def getData(self):
#    return self.data