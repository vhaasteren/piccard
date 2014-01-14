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

from PyQt4 import QtGui, QtCore

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

from plk import *

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
The Piccard main window
"""
class PiccardWidget(QtGui.QWidget):
    def __init__(self, parent=None, **kwargs):
        super(PiccardWidget, self).__init__(parent, **kwargs)

        self.parent = parent

        self.initPiccard()

    def initPiccard(self):
        print("Init Piccard")



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
        #self.hbox.addStretch(1)
        #self.hbox.addWidget(self.consoleWidget)

        self.mainFrame.setLayout(self.hbox)
        self.setCentralWidget(self.mainFrame)

    def enableConsoleWidget(self, show=True):
        if show:
            # Add, if we don't have it yet
            self.hbox.addStretch(1)
            self.hbox.addWidget(self.consoleWidget)
        else:
            # Remove, if we do have it
            pass

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

    def keyPressEvent(self, event):

        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            self.close()
        elif key == QtCore.Qt.Key_Left:
            print("Left pressed")
            self.enableConsoleWidget(True)

        else:
            print("Other key")

        
def main():
    app = QtGui.QApplication(sys.argv)
    qtipwin = QtipWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
