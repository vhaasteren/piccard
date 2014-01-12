#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
qtip: Qt interactive interface for PTA data analysis tools

"""


from __future__ import print_function
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

# Import libstempo and Piccard
try:
    import piccard as pic
except ImportError:
    pic is None
try:
    import libstempo as t2
except ImportError:
    t2 = None


# This is just to test...
def print_process_id():
    print('Process ID is:', os.getpid())

"""
Main Qtip window

Note, is the main window now, but the content will later be moved to a libstempo
tab, as part of the Piccard suite
"""
class QtipWindow(QtGui.QMainWindow):
    
    def __init__(self):
        super(QtipWindow, self).__init__()
        self.setWindowTitle('QtIpython interface to Piccard/libstempo')
        
        #self.initUI()
        self.createPlkWidget()
        self.createIPythonWidget()
        self.setPlkLayout()

        self.onDraw()
        self.show()

    def onAbout(self):
        msg = """ A demo of using PyQt with matplotlib:
        
         * Use the matplotlib navigation bar
         * Add values to the text box and press Enter (or click "Draw")
         * Show or hide the grid
         * Drag the slider to modify the width of the bars
         * Save the plot to a file using the File menu
         * Click on a bar to receive an informative message
        """
        QMessageBox.about(self, "About the demo", msg.strip())

    def createPlkWidget(self):
        self.plkWidget = QtGui.QWidget()

        # Create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #
        self.plkDpi = 100
        self.plkFig = Figure((5.0, 4.0), dpi=self.plkDpi)
        self.plkCanvas = FigureCanvas(self.plkFig)
        self.plkCanvas.setParent(self.plkWidget)

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

    def onDraw(self):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.plot(np.arange(0, 10, 0.1), np.sin(np.arange(0, 10, 0.1)), 'b-')
        self.plkCanvas.draw()

    def createIPythonWidget(self):
        #app = guisupport.get_app_qt4()

        # Create an in-process kernel
        # >>> print_process_id()
        # will print the same process ID as the main process
        self.kernelManager = QtInProcessKernelManager()
        self.kernelManager.start_kernel()
        self.kernel = self.kernelManager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push({'foo': 43, 'print_process_id': print_process_id})

        self.kernelClient = self.kernelManager.client()
        self.kernelClient.start_channels()

        #def stop():
        #    kernel_client.stop_channels()
        #    kernel_manager.shutdown_kernel()
        #    app.exit()

        self.consoleWidget = RichIPythonWidget()
        self.consoleWidget.kernel_manager = self.kernelManager
        self.consoleWidget.kernel_client = self.kernelClient
        #control.exit_requested.connect(stop)
        self.consoleWidget.set_default_style(colors='linux')
        #self.consoleWidget.show()

        #guisupport.start_event_loop_qt4(app)

    def setPlkLayout(self):
        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.plkWidget)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.consoleWidget)
        self.setLayout(self.hbox)


def main1():
    # Print the ID of the main process
    print_process_id()

    app = guisupport.get_app_qt4()

    # Create an in-process kernel
    # >>> print_process_id()
    # will print the same process ID as the main process
    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()
    kernel = kernel_manager.kernel
    kernel.gui = 'qt4'
    kernel.shell.push({'foo': 43, 'print_process_id': print_process_id})

    kernel_client = kernel_manager.client()
    kernel_client.start_channels()

    def stop():
        kernel_client.stop_channels()
        kernel_manager.shutdown_kernel()
        app.exit()

    control = RichIPythonWidget()
    control.kernel_manager = kernel_manager
    control.kernel_client = kernel_client
    control.exit_requested.connect(stop)
    control.set_default_style(colors='linux')
    control.show()

    guisupport.start_event_loop_qt4(app)



class Example(QtGui.QWidget):
    
    def __init__(self):
        super(Example, self).__init__()
        
        self.initUI()
        
    def initUI(self):
        
        QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))
        
        self.setToolTip('This is a <b>QWidget</b> widget')
        
        btn = QtGui.QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.resize(btn.sizeHint())
        btn.move(50, 50)       
        
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Tooltips')    
        self.show()
        
def main():
    app = QtGui.QApplication(sys.argv)
    qtipwin = QtipWindow()
    sys.exit(app.exec_())
    #qtipwin.start_event_loop_qt4(app)

if __name__ == '__main__':
    main()

