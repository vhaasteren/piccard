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

# For date conversions
import jdcal        # pip install jdcal

# Import libstempo and Piccard
try:
    import piccard as pic
except ImportError:
    pic is None
try:
    import libstempo as t2
except ImportError:
    t2 = None

# TODO: Implement all features with signals, rather than sharing the psr object
#       This code should not have any knowledge about libstempo. Just
#       communicate it all back to the mainFrame

"""
A widget that shows some action items, like re-fit, write par, write tim, etc.
These items are shown as buttons
"""
class PlkActionsWidget(QtGui.QWidget):
    def __init__(self, parent=None, **kwargs):
        super(PlkActionsWidget, self).__init__(parent, **kwargs)

        self.parent = parent
        self.updatePlot = None
        self.psr = None

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

    """
    We've got a pulsar
    """
    def setPulsar(self, psr, updatePlot):
        self.psr = psr
        self.updatePlot = updatePlot

    def reFit(self):
        if not self.psr is None:
            self.psr.fit()
            self.updatePlot()

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
        self.psr = None

        # The checkboxes are ordered on a grid
        self.hbox = QtGui.QHBoxLayout()     # One horizontal layout
        self.vboxes = []                    # Several vertical layouts (9 per line)
        self.fitboxPerLine = 8

        self.setPlkFitboxesLayout()


    """
    Set the layout of this widget
    """
    def setPlkFitboxesLayout(self):
        # Initialise the layout of the fit-box Widget
        # Initially there are no fitboxes, so just add the hbox
        self.setLayout(self.hbox)

    """
    We've got a new pulsar

    @param psr:     The new libstempo psr object
    """
    def setPulsar(self, psr):
        self.psr = psr
        self.deleteFitCheckBoxes()
        self.addFitCheckBoxes(psr.setpars, psr.fitpars)

    """
    Add the fitting checkboxes at the top of the plk Window

    @param setpars:     The parameters that are 'set' (in the model)
    @param fitpars:     The parameters that are currently being fitted for
    """
    def addFitCheckBoxes(self, setpars, fitpars):
        # Delete the fitboxes if there were still some left
        if not len(self.vboxes) == 0:
            self.deleteFitCheckBoxes()

        # First add all the vbox layouts
        for ii in range(min(self.fitboxPerLine, len(setpars))):
            self.vboxes.append(QtGui.QVBoxLayout())
            self.hbox.addLayout(self.vboxes[-1])

        # Then add the checkbox widgets to the vboxes
        index = 0
        for pp, par in enumerate(setpars):
            if not par in ['START', 'FINISH']:
                vboxind = index % self.fitboxPerLine

                cb = QtGui.QCheckBox(par, self)
                cb.stateChanged.connect(self.changedFitCheckBox)

                # Set checked/unchecked
                cb.setChecked(par in fitpars)

                self.vboxes[vboxind].addWidget(cb)
                index += 1

        for vv, vbox in enumerate(self.vboxes):
            vbox.addStretch(1)


    """
    Delete all the checkboxes from the Widget. Used when a new pulsar is loaded
    """
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

    """
    This is the signal handler when a checkbox is changed. The changed checkbox
    value will be propagated back to the psr object.
    """
    def changedFitCheckBox(self):
        # Check who sent the signal
        sender = self.sender()
        parchanged = sender.text()

        # Whatevs, we can just as well re-scan all the CheckButtons, and re-do
        # the fit
        for fcbox in self.vboxes:
            items = (fcbox.itemAt(i) for i in range(fcbox.count())) 
            for w in items:
                #print("Text is:", w.widget().text())
                if isinstance(w, QtGui.QWidgetItem) and \
                        isinstance(w.widget(), QtGui.QCheckBox) and \
                        parchanged == w.widget().text():
                    self.psr[parchanged].fit = bool(w.widget().checkState())
                    print("{0} set to {1}".format(parchanged, self.psr[parchanged].fit))



"""
A widget that allows one to choose which quantities to plot against each other
"""
class PlkXYPlotWidget(QtGui.QWidget):
    def __init__(self, parent=None, **kwargs):
        super(PlkXYPlotWidget, self).__init__(parent, **kwargs)

        self.psr = None
        self.updatePlotL = None
        self.parent = parent

        # We are going to use a grid layout:
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(10)

        self.xButtonGroup = QtGui.QButtonGroup(self)
        self.xButtonGroup.buttonClicked[int].connect(self.updateXPlot)
        self.yButtonGroup = QtGui.QButtonGroup(self)
        self.yButtonGroup.buttonClicked[int].connect(self.updateYPlot)

        self.xSelected = 0
        self.ySelected = 0

        # TODO: implement this:
        # Connect the 'buttonClicked' signal 'self.setLabel'
        # There are two overloads for 'buttonClicked' signal: QAbstractButton (button itself) or int (id)
        # Specific overload for the signal is selected via [QtGui.QAbstractButton]
        # Clicking any button in the QButtonGroup will send this signal with the button
        # self.buttonGroup.buttonClicked[QtGui.QAbstractButton].connect(self.setLabel)
        # def setLabel(self, button):
        self.xychoices = ['pre-fit', 'post-fit', 'date', 'orbital phase', 'siderial', \
            'day of year', 'frequency', 'TOA error', 'year', 'elevation', \
            'rounded MJD', 'sidereal time', 'hour angle', 'para. angle']
    
        self.setPlkXYPlotLayout()

    def setPlkXYPlotLayout(self):
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
            self.grid.addWidget(radio, 1+ii, labellength, 1, 1)
            if choice.lower() == 'date':
                radio.setChecked(True)
                self.xSelected = ii
            self.xButtonGroup.addButton(radio)
            self.xButtonGroup.setId(radio, ii)

            radio = QtGui.QRadioButton("")
            self.grid.addWidget(radio, 1+ii, 1+labellength, 1, 1)
            if choice.lower() == 'post-fit':
                radio.setChecked(True)
                self.ySelected = ii
            self.yButtonGroup.addButton(radio)
            self.yButtonGroup.setId(radio, ii)

        self.setLayout(self.grid)

    """
    We've got a new pulsar!
    """
    def setPulsar(self, psr, updatePlotL):
        self.psr = psr
        self.updatePlotL = updatePlotL


    """
    Given a label, get the plotting quantity
    """
    def getPlotArray(self, label):
        x = np.zeros(len(self.psr.toas()))
        des = ""

        psr = self.psr
        if label == 'pre-fit':
            x = psr.prefit.residuals * 1e6
            des = r"Pre-fit residual ($\mu$s)"
        elif label == 'post-fit':
            x = psr.residuals() * 1e6
            des = r"Post-fit residual ($\mu$s)"
        elif label == 'date':
            x = psr.toas()
            des = r"MJD"
        elif label == 'orbital phase':
            if psr['T0'].set:
                tpb = (psr.toas() - psr['T0'].val) / psr['PB'].val
            elif psr['TASC'].set:
                tpb = (psr.toas() - psr['TASC'].val) / psr['PB'].val
            else:
                print("ERROR: Neither T0 nor tasc set...")
                tpb = (psr.toas() - psr['T0'].val) / psr['PB'].val
                
            if not psr['PB'].set:
                print("WARNING: This is not a binary pulsar")
                x = np.zeros(len(psr.toas()))
            else:
                if psr['PB'].set:
                    pbdot = psr['PB'].val
                    phase = tpb % 1
                    x = phase
        elif label == 'siderial':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'day of year' or label == 'year':
            # Adjusted from plk_plug.C
            jd = psr.toas() + 2400000.5
            ijd = np.round(jd)
            fjd = (jd + 0.5) - ijd

            b = np.zeros(len(ijd))
            for ii, eijd in enumerate(ijd):
                if eijd > 2299160:
                    a = np.floor((eijd-1867216.25)/36524.25)
                    b[ii] = eijd + 1 + a - np.floor(a / 4)
                else:
                    b[ii] = eijd

            c = b + 1524

            d = np.floor((c - 122.1)/365.25)
            e = np.floor(365.25*d)
            g = np.floor((c-e)/30.6001)
            day = c-e+fjd-np.floor(30.6001*g)

            month = np.zeros(len(g))
            year = np.zeros(len(g))
            for ii, eg in enumerate(g):
                if eg < 13.5:
                    month[ii] = eg - 1
                else:
                    month[ii] = eg - 13

                if month[ii] > 2.5:
                    year[ii] = d[ii] - 4716
                else:
                    year[ii] = d[ii] - 4715
                
            jyear = np.zeros(len(year))
            jmonth = np.zeros(len(day))
            jday = np.zeros(len(day))
            mjday = np.zeros(len(day))
            doy = np.zeros(len(day))
            for ii in range(len(year)):
                # Gregorian calendar to Julian calendar
                # slaCalyd(year, month, (int)day, &retYr, &retDay, &stat);
                (temp, mjday[ii]) = jdcal.gcal2jd(year[ii], month[ii], day[ii])

                (jyear[ii], jmonth[ii], jday[ii], temp) = jdcal.jd2jcal(*jdcal.gcal2jd(year[ii], month[ii], day[ii]))

                (jyear[ii], jm, jd, temp) = jdcal.jd2jcal(temp, mjday[ii])
                (temp, doy[ii]) = jdcal.jcal2jd(jyear[ii], 1, 1)
                doy[ii] += 1
                

            # Not ok yet...
            print("WARNING: jdcal not used well yet...")

            if label == 'day of year':
                x = doy
                #print("Day of year {0}".format(x))
            else:
                #x = jyear + (jday + (day-np.floor(day)))/365.25
                x = jyear
                #print("Year {0}".format(x))
        elif label == 'frequency':
            x = self.psr.freqs
            des = r"Observing frequency (MHz)"
        elif label == 'TOA error':
            x = self.psr.toaerrs
        elif label == 'elevation':
            print("WARNING: parameter {0} not yet implemented".format(label))
            # Need to have observatory implemented in libstempo
            """
                observatory *obs;
                double source_elevation;

                obs = getObservatory(psr[0].obsn[iobs].telID);
                // get source elevation neglecting proper motion
                source_elevation = asin(dotproduct(psr[0].obsn[iobs].zenith,
                       psr[0].posPulsar)
                    / obs->height_grs80);
                x[count] = source_elevation*180.0/M_PI;
            """
        elif label == 'rounded MJD':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'sidereal time':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'hour angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'para. angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        
        return x, des

    """
    Given a label, get the plotting quantity error if there is one
    """
    def getPlotArrayErr(self, label):
        x = None
        if label == 'pre-fit':
            x = self.psr.toaerrs
        elif label == 'post-fit':
            x = self.psr.toaerrs
        elif label == 'date':
            x = self.psr.toaerrs
        elif label == 'orbital phase':
            pass
        elif label == 'siderial':
            x = self.psr.toaerrs
        elif label == 'day of year':
            pass
        elif label == 'frequency':
            pass
        elif label == 'TOA error':
            pass
        elif label == 'year':
            pass
        elif label == 'elevation':
            pass
        elif label == 'rounded MJD':
            pass
        elif label == 'sidereal time':
            x = self.psr.toaerrs
        elif label == 'hour angle':
            pass
        elif label == 'para. angle':
            pass

        return x
            

    """
    Assuming we have a pulsar, this function will return the x, y, yerr, and
    labels for the plot to be made.
    """
    def getPlot(self):
        x = None
        y = None
        yerr = None
        xlabel = None
        ylabel = None
        title = None

        if not self.psr is None and not self.updatePlotL is None:
            title = self.psr.name
            xid = self.xychoices[self.xSelected]
            yid = self.xychoices[self.ySelected]
            x, xlabel = self.getPlotArray(xid)
            y, ylabel = self.getPlotArray(yid)
            yerr = self.getPlotArrayErr(yid)

        return (x, y, yerr, xlabel, ylabel, title)

    """
    The x-plot radio buttons have been updated
    """
    def updateXPlot(self, newid):
        self.xSelected = newid #-2 - newid     # WTF????
        self.updateChoice()

    """
    The y-plot radio buttons have been updated
    """
    def updateYPlot(self, newid):
        self.ySelected = newid #-2 - newid     # WTF? What's with the index??
        self.updateChoice()

    def updateChoice(self):
        # updatePLot was the callback from the main widget
        if not self.updatePlotL is None:
            self.updatePlotL(*self.getPlot())



"""
The plk-emulator window.
"""
class PlkWidget(QtGui.QWidget):

    def __init__(self, parent=None, **kwargs):
        super(PlkWidget, self).__init__(parent, **kwargs)

        self.initPlk()
        self.initPlkLayout()
        self.showVisibleWidgets()

        self.psr = None
        self.parent = parent

    def initPlk(self):
        self.setMinimumSize(650, 550)

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

        # At startup, all the widgets are visible
        self.xyChoiceVisible = True
        self.fitboxVisible = True
        self.actionsVisible = True
        self.layoutMode = 1         # (0 = none, 1 = all, 2 = only fitboxes, 3 = fit & action)


    """
    When we don't have a pulsar yet, but we have to display something, just draw
    an empty figure
    """
    def drawSomething(self):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.set_xlabel('MJD')
        self.plkAxes.set_ylabel('Residual ($\mu$s)')
        self.plkCanvas.draw()

    """
    We've got a new pulsar!
    """
    def setPulsar(self, psr):
        self.psr = psr

        # Update the fitting checkboxes
        self.fitboxesWidget.setPulsar(psr)
        self.xyChoiceWidget.setPulsar(psr, self.updatePlotL)
        self.actionsWidget.setPulsar(psr, self.updatePlot)

        # Draw the residuals
        self.xyChoiceWidget.updateChoice()
        self.show()

    """
    This function is called when we have new fitparameters

    TODO: callback not used right now
    """
    def newFitParameters(self):
        pass

    """
    Initialise the basic layout of this plk emulator emulator
    """
    def initPlkLayout(self):
        # Initialise the plk box
        self.plkbox.addWidget(self.fitboxesWidget)

        self.xyplotbox.addWidget(self.xyChoiceWidget)
        self.xyplotbox.addWidget(self.plkCanvas)

        self.plkbox.addLayout(self.xyplotbox)

        self.plkbox.addWidget(self.actionsWidget)
        self.setLayout(self.plkbox)

    """
    Show the correct widgets in the plk Window
    """
    def showVisibleWidgets(self):
        self.xyChoiceWidget.setVisible(self.xyChoiceVisible)
        self.fitboxesWidget.setVisible(self.fitboxVisible)
        self.actionsWidget.setVisible(self.actionsVisible)


    """
    Update the plot, without having all the information
    """
    def updatePlot(self):
        if not self.psr is None:
            self.updatePlotL(*self.xyChoiceWidget.getPlot())

    """
    Update the plot, given all the plotting info
    """
    def updatePlotL(self, x, y, yerr, xlabel, ylabel, title):
        self.plkAxes.clear()
        self.plkAxes.grid(True)

        xave = 0.5 * (np.max(x) + np.min(x))
        xmin = xave - 1.05 * (xave - np.min(x))
        xmax = xave + 1.05 * (np.max(x) - xave)
        if yerr is None:
            yave = 0.5 * (np.max(y) + np.min(y))
            ymin = yave - 1.05 * (yave - np.min(y))
            ymax = yave + 1.05 * (np.max(y) - yave)
            self.plkAxes.scatter(x, y, marker='.', c='g')
        else:
            yave = 0.5 * (np.max(y+yerr) + np.min(y-yerr))
            ymin = yave - 1.05 * (yave - np.min(y-yerr))
            ymax = yave + 1.05 * (np.max(y+yerr) - yave)
            self.plkAxes.errorbar(x, y, yerr=yerr, fmt='.', color='green')

        self.plkAxes.axis([xmin, xmax, ymin, ymax])
        self.plkAxes.set_xlabel(xlabel)
        self.plkAxes.set_ylabel(ylabel)
        self.plkAxes.set_title(title)
        self.plkCanvas.draw()


    """
    A key is pressed. Handle all the shortcuts here
    """
    def keyPressEvent(self, event):

        key = event.key()
        modifiers = int(event.modifiers())

        #if int(e.modifiers()) == (QtCore.Qt.ControlModifier+QtCore.Qt.AltModifier)

        if key == QtCore.Qt.Key_Escape:
            if self.parent is None:
                self.close()
            else:
                self.parent.close()
        elif (key == ord('M') or key == ord('m')) and \
                modifiers == QtCore.Qt.ControlModifier:
            # Change the window
            self.layoutMode = (1+self.layoutMode)%4
            if self.layoutMode == 0:
                self.xyChoiceVisible = False
                self.fitboxVisible = False
                self.actionsVisible = False
            elif self.layoutMode == 1:
                self.xyChoiceVisible = True
                self.fitboxVisible = True
                self.actionsVisible = True
            elif self.layoutMode == 2:
                self.xyChoiceVisible = False
                self.fitboxVisible = True
                self.actionsVisible = True
            elif self.layoutMode == 3:
                self.xyChoiceVisible = False
                self.fitboxVisible = True
                self.actionsVisible = False
            self.showVisibleWidgets()

        elif key == QtCore.Qt.Key_Left:
            # print("Left pressed")
            pass
        else:
            #print("Other key: {0} {1} {2} {3}".format(key,
            #    modifiers, ord('M'), QtCore.Qt.ControlModifier))
            pass
