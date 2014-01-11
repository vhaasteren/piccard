from Tkinter import *
import sys
import time
import mod1

old_stdout = sys.stdout

class StdoutRedirector(object):

    def __init__(self, text_area):
        self.text_area = text_area

    def write(self, str):
        self.text_area.insert(END, str)
        self.text_area.see(END)

def sleep():
    print('Received sleep-command')
    sleepBtn.update_idletasks()         # To update the screen
    # sys.stdout.flush()                  # To update the stdout? No, does not work
    time.sleep(2)
    mod1.main()
    #sleepBtn.update_idletasks()         # To update the screen

root = Tk()

# Textbox
outputPanel = Text(root, wrap='word', height = 11, width=50)
outputPanel.grid(column=0, row=0, columnspan = 2, sticky='NSWE', padx=5, pady=5)
sys.stdout = StdoutRedirector(outputPanel)

# Sleep button
sleepBtn = Button(root, text='Sleep', command=sleep)
sleepBtn.grid(row=1, column=1, sticky='E', padx=5, pady=5)

root.mainloop()

sys.stdout = old_stdout
