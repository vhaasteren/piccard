#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ZetCode Tkinter tutorial

This program creates a quit
button. When we press the button,
the application terminates. 

author: Jan Bodnar
last modified: December 2010
website: www.zetcode.com
"""

from Tkinter import Tk, BOTH
from ttk import Frame, Button, Style


class Example(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent)   
         
        self.parent = parent
        
        self.initUI()
        
    def initUI(self):
      
        self.parent.title("Quit button")
        self.style = Style()
        self.style.theme_use("default")

        self.pack(fill=BOTH, expand=1)

        quitButton = Button(self, text="Quit",
            command=self.quit)
        quitButton.place(x=50, y=50)


def main():
  
    root = Tk()
    root.geometry("250x150+300+300")
    app = Example(root)
    root.mainloop()  


if __name__ == '__main__':
    main()
