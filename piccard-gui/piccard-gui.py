#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
piccard-gui.py

A Tkinter user-interface to the piccard functionality
"""

from __future__ import division

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import numpy as np
import piccard as pic
import sys
import ttk

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


# Derive the main class from ttk.Notebook. It will be a tabbed interface
class PicMainWindow(object, ttk.Notebook):
    def __init__(self, master=None, **kw):
        npages = kw.pop('npages', 3)            # Default number of pages is 3

        # Minimally one page
        if npages < 1:
            npages = 1

        #kw['style'] = 'PicMainWindow.TNotebook'
        #ttk.Style(master).layout('PicMainWindow.TNotebook.Tab', '')
        ttk.Notebook.__init__(self, master, **kw)

        self._children = []

        self.pack(fill='both', expand=1) 

        # Add the first page
        c = Tk.Canvas(self) 
        self.add(c, text='bar') 
        self._children.append(c)
        self.add_plot()

        # Fill the text areas
        for page in range(1, npages):
            t = Tk.Text(self)
            self.add(t, text='page-'+str(page))
            self._children.append(t)
            #self.add_empty_page()

    def add_plot(self):
        # Make a simple plot
        x = np.arange(0, 30, 0.1)
        y = np.sin(x)

        # Make the matplotlib plot
        f = Figure(figsize=(4,3), dpi=100)
        a = f.add_subplot(111)
        a.plot(x, y, 'b-')
        a.grid(True)
        a.set_title('Tk embedding')
        a.set_xlabel('X axis label')
        a.set_ylabel('Y label')

        # Embed the plot on a canvas
        canvas = FigureCanvasTkAgg(f, master=self._children[0])
        canvas.show()

        # Pack describes how the widget will be place
        # side: TOP, BOTTOM, LEFT, RIGHT
        # fill: NONE, X, Y, BOTH
        # expand: True, False
        canvas.get_tk_widget().pack(side=Tk.LEFT, fill=Tk.NONE, expand=False)

        #button = Tk.Button(master=self._children[0], text='Quit', command=sys.exit)
        #button.pack(side=Tk.BOTTOM)



# Derive the main class from ttk.Notebook. It will be a tabbed interface
class MainWindowWiz(object, ttk.Notebook):
    def __init__(self, master=None, **kw):
        npages = kw.pop('npages', 3)
        kw['style'] = 'MainWindowWiz.TNotebook'
        ttk.Style(master).layout('MainWindowWiz.TNotebook.Tab', '')
        ttk.Notebook.__init__(self, master, **kw)

        self._children = {}

        for page in range(npages):
            self.add_empty_page()

        self.current = 0
        self._wizard_buttons()

    def _wizard_buttons(self):
        """Place wizard buttons in the pages."""
        for indx, child in self._children.iteritems():
            btnframe = ttk.Frame(child)
            btnframe.pack(side='bottom', fill='x', padx=6, pady=12)

            nextbtn = ttk.Button(btnframe, text="Next", command=self.next_page)
            nextbtn.pack(side='right', anchor='e', padx=6)
            if indx != 0:
                prevbtn = ttk.Button(btnframe, text="Previous",
                    command=self.prev_page)
                prevbtn.pack(side='right', anchor='e', padx=6)

                if indx == len(self._children) - 1:
                    nextbtn.configure(text="Finish", command=self.close)

    def next_page(self):
        self.current += 1

    def prev_page(self):
        self.current -= 1

    def close(self):
        self.master.destroy()

    def add_empty_page(self):
        child = ttk.Frame(self)
        self._children[len(self._children)] = child
        self.add(child)

    def add_page_body(self, body):
        body.pack(side='top', fill='both', padx=6, pady=12)

    def page_container(self, page_num):
        if page_num in self._children:
            return self._children[page_num]
        else:
            raise KeyError("Invalid page: %s" % page_num)

    def _get_current(self):
        return self._current
    
    def _set_current(self, curr):
        if curr not in self._children:
            raise KeyError("Invalid page: %s" % curr)

        self._current = curr
        self.select(self._children[self._current])

    current = property(_get_current, _set_current)


def demo():
    root = Tk.Tk()
    wizard = MainWindowWiz(npages=3)
    wizard.master.minsize(400, 350)
    page0 = ttk.Label(wizard.page_container(0), text='Page 1')
    page1 = ttk.Label(wizard.page_container(1), text='Page 2')
    page2 = ttk.Label(wizard.page_container(2), text='Page 3')
    wizard.add_page_body(page0)
    wizard.add_page_body(page1)
    wizard.add_page_body(page2)
    wizard.pack(fill='both', expand=True)
    root.mainloop()

def demo2():
    root = Tk.Tk()
    picwin = PicMainWindow(root, npages=4)
    picwin.master.minsize(800, 500)
    root.mainloop()

if __name__ == "__main__":
    demo2()
