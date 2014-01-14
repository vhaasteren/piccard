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


"""
A widget that shows some action items, like re-fit, write par, write tim, etc.
These items are shown as buttons
"""
class PlkActionsWidget(QtGui.QWidget):
    def __init__(self, parent=None, **kwargs):
        super(PlkActionsWidget, self).__init__(parent, **kwargs)

        self.parent = parent

        self.hbox = QtGui.QHBoxLayout()     # One horizontal layout

        self.setPlkActionsWidget()

    def setPlkActionsWidget(self):
        button = QtGui.QPushButton('Re-fit')
        button.clicked.connect(self.reFit)
        self.hbox.addWidget(button)

        button = QtGui.QPushButton('Clear')
        button.clicked.connect(self.clearAll)
        self.hbox.addWidget(button)

        button = QtGui.QPushButton('Write par')
        button.clicked.connect(self.writePar)
        self.hbox.addWidget(button)

        button = QtGui.QPushButton('Write tim')
        button.clicked.connect(self.writeTim)
        self.hbox.addWidget(button)

        button = QtGui.QPushButton('Save fig')
        button.clicked.connect(self.saveFig)
        self.hbox.addWidget(button)

        self.hbox.addStretch(1)

        self.setLayout(self.hbox)

    def reFit(self):
        print("Re-fit clicked")

    def writePar(self):
        print("Write Par clicked")

    def writeTim(self):
        print("Write Tim clicked")

    def clearAll(self):
        print("Clear clicked")

    def saveFig(self):
        print("Save fig clicked")


"""
A widget that allows one to select which parameters to fit for
"""
class PlkFitboxesWidget(QtGui.QWidget):
    def __init__(self, parent=None, **kwargs):
        super(PlkFitboxesWidget, self).__init__(parent, **kwargs)

        self.parent = parent

        # The checkboxes are ordered on a grid
        self.hbox = QtGui.QHBoxLayout()     # One horizontal layout
        self.vboxes = []                    # Several vertical layouts (9 per line)
        self.fitboxPerLine = 9

        self.setPlkFitboxesLayout()


    def setPlkFitboxesLayout(self):
        # Initialise the layout of the fit-box Widget
        # Initially there are no fitboxes, so just add the hbox
        self.setLayout(self.hbox)

    def addFitCheckBoxes(self, pars):
        # Delete the fitboxes if there were still some left
        if not len(self.vboxes) == 0:
            self.deleteFitCheckBoxes()

        # First add all the vbox layouts
        for ii in range(min(self.fitboxPerLine, len(pars))):
            self.vboxes.append(QtGui.QVBoxLayout())
            self.hbox.addLayout(self.vboxes[-1])

        # Then add the checkbox widgets to the vboxes
        for pp, par in enumerate(pars):
            vboxind = pp % self.fitboxPerLine

            cb = QtGui.QCheckBox(par, self)
            cb.stateChanged.connect(self.changedFitCheckBox)

            self.vboxes[vboxind].addWidget(cb)

        for vv, vbox in enumerate(self.vboxes):
            vbox.addStretch(1)


    def deleteFitCheckBoxes(self):
        for fcbox in self.vboxes:
            while fcbox.count():
                item = fcbox.takeAt(0)
                if isinstance(item, QtGui.QWidgetItem):
                    item.widget().deleteLater()
                elif isinstance(item, QtGui.QSpacerItem):
                    fcbox.removeItem(item)
                else:
                    fcbox.clearLayout(item.layout())
                    fcbox.removeItem(item)


        for fcbox in self.vboxes:
            self.hbox.removeItem(fcbox)

        self.vboxes = []

    def changedFitCheckBox(self):
        # Check who sent the signal
        sender = self.sender()
        parchanged = sender.text()

        # Whatevs, we can just as well re-scan all the CheckButtons, and re-do
        # the fit
        # TODO: not implemented
        print("Checkbox", parchanged, "changed")


"""
A widget that allows one to choose which quantities to plot against each other
"""
class PlkXYPlotWidget(QtGui.QWidget):
    def __init__(self, parent=None, **kwargs):
        super(PlkXYPlotWidget, self).__init__(parent, **kwargs)

        self.parent = parent

        # We are going to use a grid layout:
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(10)

        self.xButtonGroup = QtGui.QButtonGroup(self)
        self.yButtonGroup = QtGui.QButtonGroup(self)
        #self.xButtonGroup.onclicked.connect(self.updateChoice)
        #self.yButtonGroup.onclicked.connect(self.updateChoice)
    
        self.setPlkXYPlotLayout()

    def setPlkXYPlotLayout(self):
        self.xychoices = ['pre-fit', 'post-fit', 'date', 'orbital phase', 'siderial', \
            'day of year', 'frequency', 'TOA error', 'year', 'elevation', \
            'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']

        labellength = 3

        label = QtGui.QLabel(self)
        label.setText("")
        self.grid.addWidget(label, 0, 0, 1, labellength)
        label = QtGui.QLabel(self)
        label.setText("X")
        self.grid.addWidget(label, 0, 0+labellength, 1, 1)
        label = QtGui.QLabel(self)
        label.setText("Y")
        self.grid.addWidget(label, 0, 1+labellength, 1, 1)

        # Add all the xychoices
        for ii, choice in enumerate(self.xychoices):
            # The label of the choice
            label = QtGui.QLabel(self)
            label.setText(choice)
            self.grid.addWidget(label, 1+ii, 0, 1, labellength)

            # The X and Y radio buttons
            radio = QtGui.QRadioButton("")
            radio.toggled.connect(self.updateChoice)
            self.grid.addWidget(radio, 1+ii, labellength, 1, 1)
            self.xButtonGroup.addButton(radio)
            #if choice.lower() == 'date':
            #    radio.SetChecked(True)

            radio = QtGui.QRadioButton("")
            radio.toggled.connect(self.updateChoice)
            self.grid.addWidget(radio, 1+ii, 1+labellength, 1, 1)
            self.yButtonGroup.addButton(radio)
            #if choice.lower() == 'post-fit':
            #    radio.SetChecked(True)

        self.setLayout(self.grid)

    def updateChoice(self):
        print("update Choice!")



"""
The plk-emulator window.
"""
class PlkWidget(QtGui.QWidget):

    def __init__(self, parent=None, **kwargs):
        super(PlkWidget, self).__init__(parent, **kwargs)

        self.initPlk()
        self.setPlkLayout()

        self.psr = None
        self.parent = parent

    def initPlk(self):
        self.setMinimumSize(650, 500)

        self.plkbox = QtGui.QVBoxLayout()                       # plkbox contains the whole plk widget
        self.xyplotbox = QtGui.QHBoxLayout()                    # plkbox contains the whole plk widget
        self.fitboxesWidget = PlkFitboxesWidget(parent=self)    # Contains all the checkboxes
        self.actionsWidget = PlkActionsWidget(parent=self)

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

        # Create the XY choice widget
        self.xyChoiceWidget = PlkXYPlotWidget(parent=self)


    def drawSomething(self):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkCanvas.draw()

    def setPulsar(self, psr):
        # Update the fitting checkboxes
        self.fitboxesWidget.deleteFitCheckBoxes()
        self.fitboxesWidget.addFitCheckBoxes(psr.pars)

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
        self.plkbox.addWidget(self.fitboxesWidget)

        self.xyplotbox.addWidget(self.xyChoiceWidget)
        self.xyplotbox.addWidget(self.plkCanvas)

        #self.plkbox.addWidget(self.plkCanvas)
        self.plkbox.addLayout(self.xyplotbox)

        self.plkbox.addWidget(self.actionsWidget)
        self.setLayout(self.plkbox)

    def keyPressEvent(self, event):

        key = event.key()

        if key == QtCore.Qt.Key_Escape:
            if self.parent is None:
                self.close()
            else:
                self.parent.close()
        elif key == QtCore.Qt.Key_Left:
            print("Left pressed")

        else:
            print("Other key")
