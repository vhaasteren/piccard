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
        print("Hello, Piccard!")

"""
The open-something Widget. First shown in the main window, if it is not started
with a command to open any file to begin with. This way, we don't have to show
an empty graph
"""
class OpenSomethingWidget(QtGui.QWidget):
    def __init__(self, parent=None, openFile=None, **kwargs):
        super(OpenSomethingWidget, self).__init__(parent, **kwargs)

        self.parent = parent
        self.openFileFn = openFile

        self.initOSWidget()

    """
    Initialise the widget with a button and a label
    """
    def initOSWidget(self):
        self.vbox = QtGui.QVBoxLayout()

        button = QtGui.QPushButton("Open a file...")
        button.clicked.connect(self.openFile)
        self.vbox.addWidget(button)

        self.setLayout(self.vbox)

    """
    Display the open file dialog, and send the parent window the order to open
    the file
    """
    def openFile(self):
        if not self.openFileFn is None:
            filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '~/')
            self.openFileFn(filename)



"""
Main Qtip window

Note, is the main window now, but the content will later be moved to a libstempo
tab, as part of the Piccard suite
"""
class QtipWindow(QtGui.QMainWindow):
    
    def __init__(self, parent=None):
        super(QtipWindow, self).__init__(parent)
        self.setWindowTitle('QtIpython interface to Piccard/libstempo')
        
        # Initialise basic gui elements
        self.initUI()

        # Start the embedded IPython kernel
        self.createIPythonKernel()

        # Create the display widgets
        self.createPlkWidget()
        self.createIPythonWidget()
        self.createOpenSomethingWidget()

        self.initQtipLayout()
        self.setQtipLayout(whichWidget='opensomething', showIPython=False)

        self.show()

    def __del__(self):
        pass

    def onAbout(self):
        msg = """ A demo of using PyQt with matplotlib, libstempo, and IPython:
        """
        QtGui.QMessageBox.about(self, "About the demo", msg.strip())

    """
    Initialise the user-interface elements
    """
    def initUI(self):
        # Create the main-frame widget, and the layout
        self.mainFrame = QtGui.QWidget()
        self.setCentralWidget(self.mainFrame)
        self.hbox = QtGui.QHBoxLayout()     # HBox contains all widgets

        # Create the menu action items
        self.openParTimAction = QtGui.QAction('&Open', self)        
        self.openParTimAction.setShortcut('Ctrl+O')
        self.openParTimAction.setStatusTip('Open par/tim')
        self.openParTimAction.triggered.connect(self.openParTim)

        self.exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.exitAction.triggered.connect(self.close)

        self.toggleIPythonAction = QtGui.QAction('&IPython', self)        
        self.toggleIPythonAction.setShortcut('Ctrl+I')
        self.toggleIPythonAction.setStatusTip('Toggle IPython')
        self.toggleIPythonAction.triggered.connect(self.toggleIPython)

        self.aboutAction = QtGui.QAction('&About', self)        
        self.aboutAction.setShortcut('Ctrl+A')
        self.aboutAction.setStatusTip('About Qtip')
        self.aboutAction.triggered.connect(self.onAbout)

        self.statusBar()
        
        # On OSX, make sure the menu can be displayed (in the window itself)
        if sys.platform == 'darwin':
            # On OSX, the menubar is usually on the top of the screen, not in
            # the window. To make it in the window:
            QtGui.qt_mac_set_native_menubar(False) 

            # Otherwise, if we'd like to get the system menubar at the top, then
            # we need another menubar object, not self.menuBar as below. In that
            # case, use:
            # self.menubar = QtGui.QMenuBar()
            # TODO: Somehow this does not work. Per-window one does though

        # Create the menu
        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&File')
        self.fileMenu.addAction(self.openParTimAction)
        self.fileMenu.addAction(self.exitAction)
        self.viewMenu = self.menubar.addMenu('&View')
        self.viewMenu.addAction(self.toggleIPythonAction)
        self.helpMenu = self.menubar.addMenu('&Help')
        self.helpMenu.addAction(self.aboutAction)

        # What is the status quo of the user interface?
        self.showIPython = False
        self.whichWidget = 'None'
        self.prevShowIPython = False
        self.prevWhichWidget = 'None'

    """
    Create the IPython Kernel
    """
    def createIPythonKernel(self):
        # Create an in-process kernel
        self.kernelManager = QtInProcessKernelManager()
        self.kernelManager.start_kernel()
        self.kernel = self.kernelManager.kernel

        self.kernelClient = self.kernelManager.client()
        self.kernelClient.start_channels()

        self.kernel.shell.enable_matplotlib(gui='inline')

        # Load the necessary packages in the embedded kernel
        cell = "import numpy as np, matplotlib.pyplot as plt, libstempo as t2"
        self.kernel.shell.run_cell(cell)


    """
    Create the OpenSomething widget. Do not add it to the layout yet

    TODO:   probably should use a signal to implement this call instead of
            openParTim
    """
    def createOpenSomethingWidget(self):
        self.openSomethingWidget = OpenSomethingWidget(parent=self.mainFrame, \
                openFile=self.requestOpenPlk)
        self.openSomethingWidget.hide()

    """
    Create the Plk widget
    """
    def createPlkWidget(self):
        self.plkWidget = PlkWidget(parent=self.mainFrame)
        self.plkWidget.hide()


    """
    Create the IPython widget
    """
    def createIPythonWidget(self):
        self.consoleWidget = RichIPythonWidget()
        self.consoleWidget.setMinimumSize(600, 500)
        self.consoleWidget.banner = QtipBanner
        self.consoleWidget.kernel_manager = self.kernelManager
        self.consoleWidget.kernel_client = self.kernelClient
        self.consoleWidget.exit_requested.connect(self.close)
        self.consoleWidget.set_default_style(colors='linux')
        self.consoleWidget.hide()

    """
    Toggle the IPython widget on or off
    """
    def toggleIPython(self):
        self.showIPython = not self.showIPython

        self.setQtipLayout()

    """
    Initialise the Qtip layout
    """
    def initQtipLayout(self):
        self.mainFrame.setMinimumSize(650, 500)
        self.hbox.addWidget(self.openSomethingWidget)
        self.hbox.addWidget(self.plkWidget)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.consoleWidget)
        self.mainFrame.setLayout(self.hbox)

    """
    Hide all widgets of the mainFrame
    """
    def hideAllWidgets(self):
        # Remove all widgets from the main window
        # ???
        """
        while self.hbox.count():
            item = self.hbox.takeAt(0)
            if isinstance(item, QtGui.QWidgetItem):
                #item.widget().deleteLater()
                item.widget().hide()
            elif isinstance(item, QtGui.QSpacerItem):
                #self.hbox.removeItem(item)
                pass
            else:
                #fcbox.clearLayout(item.layout())
                #self.hbox.removeItem(item)
                pass
        """
        self.openSomethingWidget.hide()
        self.plkWidget.hide()
        self.consoleWidget.hide()

    """
    Show the correct widgets in the mainFrame
    """
    def showVisibleWidgets(self):
        # Add the widgets we need
        if self.whichWidget.lower() == 'opensomething':
            self.openSomethingWidget.show()
        elif self.whichWidget.lower() == 'plk':
            self.plkWidget.show()
        elif self.whichWidget.lower() == 'piccard':
            pass

        if self.showIPython:
            self.mainFrame.setMinimumSize(1300, 500)
            self.consoleWidget.show()
        else:
            self.mainFrame.setMinimumSize(650, 500)

    """
    Given which widgets to show, display the right widgets and hide the rest

    @param whichWidget:     Which widget to show
    @param showIPython:     Whether to show the IPython console
    """
    def setQtipLayout(self, whichWidget=None, showIPython=None):
        if not whichWidget is None:
            self.whichWidget = whichWidget
        if not showIPython is None:
            self.showIPython = showIPython

        # After hiding the widgets, wait 25 miliseconds before showing them again
        self.hideAllWidgets()
        QtCore.QTimer.singleShot(250, self.showVisibleWidgets)

        self.prevWhichWidget = self.whichWidget
        self.prevShowIPython = self.showIPython

        self.mainFrame.setLayout(self.hbox)
        #print("whichWidget = {0},  showIPython = {1}".format(self.whichWidget, self.showIPython))
        self.mainFrame.show()

    def enableConsoleWidget(self, show=True):
        if show:
            # Add, if we don't have it yet
            self.hbox.addStretch(1)
            self.hbox.addWidget(self.consoleWidget)
        else:
            # Remove, if we do have it
            pass

    """
    Request to open a file in the plk widget

    @param filename:    If we already know the name of the parfile, this is it
    """
    def requestOpenPlk(self, filename=None):
        self.setQtipLayout(whichWidget='plk', showIPython=self.showIPython)
        if filename is None:
            self.openParTim()
        else:
            self.openTim(filename)

    """
    Open a par-file and a tim-file
    """
    def openParTim(self, filename=None):
        print("openParTim called with {0}".format(filename))

        # Ask the user for a par and tim file, and open these with libstempo
        if isinstance(filename, str):
            parfilename = filename
        else:
            parfilename = QtGui.QFileDialog.getOpenFileName(self, 'Open par-file', '~/')

        self.openTim(parfilename)

    """
    Open a tim-file, given a par file
    """
    def openTim(self, parfilename):
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
            #self.enableConsoleWidget(True)

        else:
            print("Other key")

        
def main():
    app = QtGui.QApplication(sys.argv)
    qtipwin = QtipWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
