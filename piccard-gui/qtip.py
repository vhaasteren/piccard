#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
qtip: Qt interactive interface for PTA data analysis tools

"""


from __future__ import print_function
from __future__ import division
import os, sys

# Importing all the stuff for the IPython console widget
from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

from PyQt4 import QtGui

# Importing all the stuff for the matplotlib widget
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

# Numpy etc.
import numpy as np
import time

# Import libstempo and Piccard
try:
    import piccard as pic
except ImportError:
    pic is None
try:
    import libstempo as t2
except ImportError:
    t2 = None

# The startup banner
QtipBanner = """Qtip python console, by Rutger van Haasteren
Console powered by IPython
Type "copyright", "credits" or "license" for more information.

?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.
%guiref   -> A brief reference about the graphical user interface.

import numpy as np, matplotlib.pyplot as plt, libstempo as t2
"""


"""
The plk-emulator window.
"""
class PlkWidget(QtGui.QWidget):

    def __init__(self, parent=None, **kwargs):
        super(PlkWidget, self).__init__(parent)

        self.initPlk()
        self.setPlkLayout()

        self.psr = None
        self.parent = parent

    def initPlk(self):
        self.setMinimumSize(650, 500)

        self.plkbox = QtGui.QVBoxLayout()   # VBox contains the plk widget
        self.fcboxes = []                   # All the checkbox layouts (7 per line)
        self.fitboxPerLine = 9

        # Create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #
        self.plkDpi = 100
        self.plkFig = Figure((5.0, 4.0), dpi=self.plkDpi)
        self.plkCanvas = FigureCanvas(self.plkFig)
        self.plkCanvas.setParent(self)

        # Since we have only one plot, we can use add_axes 
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.plkAxes = self.plkFig.add_subplot(111)
        
        # Bind the 'pick' event for clicking on one of the bars
        #
        #self.canvas.mpl_connect('pick_event', self.on_pick)

        # Create the navigation toolbar, tied to the canvas
        #
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)


        # Draw an empty graph
        self.drawSomething()

    def deleteFitCheckBoxes(self):
        for fcbox in self.fcboxes:
            while fcbox.count():
                item = fcbox.takeAt(0)
                item.widget().deleteLater()

        for fcbox in self.fcboxes:
            self.plkbox.removeItem(fcbox)

        self.fcboxes = []

    def addFitCheckBoxes(self, psr):
        for pp, par in enumerate(psr.pars):
            if pp % self.fitboxPerLine == 0:
                self.fcboxes.append(QtGui.QHBoxLayout())

            cb = QtGui.QCheckBox(par, self)
            cb.stateChanged.connect(self.changedFitCheckBox)

            fcbox = self.fcboxes[-1]
            fcbox.addWidget(cb)

        for fcbox in self.fcboxes:
            #fcbox.addStretch(1)
            self.plkbox.addLayout(fcbox)

    def changedFitCheckBox(self):
        # Check who sent the signal
        sender = self.sender()
        parchanged = sender.text()

        # Whatevs, we can just as well re-scan all the CheckButtons, and re-do
        # the fit
        # TODO: not implemented
        print("Checkbox", parchanged, "changed")


    def drawSomething(self):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkCanvas.draw()

    def setPulsar(self, psr):
        # Update the fitting checkboxes
        self.deleteFitCheckBoxes()
        self.addFitCheckBoxes(psr)

        # Draw the residuals
        self.drawResiduals(psr.toas(), psr.residuals(), psr.toaerrs*1.0e-6, psr.name)
        self.psr = psr
        self.show()

    def drawResiduals(self, x, y, yerr, title=""):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.errorbar(x, y*1.0e6, yerr=yerr*1.0e6, fmt='.', color='green')
        self.plkAxes.set_xlabel(r'MJD')
        self.plkAxes.set_ylabel(r'Residual ($\mu$s)')
        self.plkAxes.set_title(title)
        self.plkCanvas.draw()

    def setPlkLayout(self):
        # Initialise the plk box
        self.plkbox.addWidget(self.plkCanvas)

        # Just in case, set all the fit-checkboxes. These are not supposed to be
        # there, by the way
        for fcbox in self.fcboxes:
            self.plkbox.addLayout(fcbox)
        self.setLayout(self.plkbox)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            if self.parent is None:
                self.close()
            else:
                self.parent.close()

"""
Main Qtip window

Note, is the main window now, but the content will later be moved to a libstempo
tab, as part of the Piccard suite
"""
class QtipWindow(QtGui.QMainWindow):
    
    def __init__(self, parent=None):
        super(QtipWindow, self).__init__(parent)
        self.setWindowTitle('QtIpython interface to Piccard/libstempo')
        
        self.initUI()
        self.createPlkWidget()
        self.createIPythonWidget()
        self.setQtipLayout()

        self.show()

    def __del__(self):
        pass

    def onAbout(self):
        msg = """ A demo of using PyQt with matplotlib, libstempo, and IPython:
        """
        QtGui.QMessageBox.about(self, "About the demo", msg.strip())

    def initUI(self):
        # The main screen. Do we need this?
        self.mainFrame = QtGui.QWidget()
        self.hbox = QtGui.QHBoxLayout()     # HBox contains both IPython and plk

        self.openParTimAction = QtGui.QAction('&Open', self)        
        self.openParTimAction.setShortcut('Ctrl+O')
        self.openParTimAction.setStatusTip('Open par/tim')
        self.openParTimAction.triggered.connect(self.openParTim)

        self.exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.exitAction.triggered.connect(self.close)

        self.aboutAction = QtGui.QAction('&About', self)        
        self.aboutAction.setShortcut('Ctrl+A')
        self.aboutAction.setStatusTip('About Qtip')
        self.aboutAction.triggered.connect(self.onAbout)

        self.statusBar()

        
        if sys.platform == 'darwin':
            # On OSX, the menubar is usually on the top of the screen, not in
            # the window. To make it in the window:
            QtGui.qt_mac_set_native_menubar(False) 

            # Otherwise, if we'd like to get the system menubar at the top, then
            # we need another menubar object, not self.menuBar as below. In that
            # case, use:
            # self.menubar = QtGui.QMenuBar()
            # TODO: Somehow this does not work. Per-window one does though

        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&File')
        self.fileMenu.addAction(self.openParTimAction)
        self.fileMenu.addAction(self.exitAction)
        self.helpMenu = self.menubar.addMenu('&Help')
        self.helpMenu.addAction(self.aboutAction)

    def createPlkWidget(self):
        self.plkWidget = PlkWidget(parent=self.mainFrame)

    def createIPythonWidget(self):
        # Create an in-process kernel
        self.kernelManager = QtInProcessKernelManager()
        self.kernelManager.start_kernel()
        self.kernel = self.kernelManager.kernel

        self.kernelClient = self.kernelManager.client()
        self.kernelClient.start_channels()

        self.consoleWidget = RichIPythonWidget()
        self.consoleWidget.setMinimumSize(600, 500)
        self.consoleWidget.banner = QtipBanner
        self.consoleWidget.kernel_manager = self.kernelManager
        self.consoleWidget.kernel_client = self.kernelClient
        self.consoleWidget.exit_requested.connect(self.close)
        self.consoleWidget.set_default_style(colors='linux')

        self.kernel.shell.enable_matplotlib(gui='inline')

        # Load the necessary packages in the embedded kernel
        cell = "import numpy as np, matplotlib.pyplot as plt, libstempo as t2"
        self.kernel.shell.run_cell(cell)

    def setQtipLayout(self):
        self.hbox.addWidget(self.plkWidget)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.consoleWidget)

        self.mainFrame.setLayout(self.hbox)
        self.setCentralWidget(self.mainFrame)

    def openParTim(self):
        # Ask the user for a par and tim file, and open these with libstempo
        parfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open par-file', '~/')
        timfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open tim-file', '~/')

        # Load the pulsar
        cell = "psr = t2.tempopulsar('"+parfilename+"', '"+timfilename+"')"
        self.kernel.shell.run_cell(cell)
        psr = self.kernel.shell.ns_table['user_local']['psr']

        # Update the plk widget
        self.plkWidget.setPulsar(psr)

        # Communicating with the kernel goes as follows
        # self.kernel.shell.push({'foo': 43, 'print_process_id': print_process_id}, interactive=True)
        # print("Embedded, we have:", self.kernel.shell.ns_table['user_local']['foo'])

        
def main():
    app = QtGui.QApplication(sys.argv)
    qtipwin = QtipWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
