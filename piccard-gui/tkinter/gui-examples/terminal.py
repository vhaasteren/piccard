import Tkinter as Tk
import os

root = Tk.Tk()
termf = Tk.Frame(root, height=400, width=500)

termf.pack(fill=Tk.BOTH, expand=Tk.YES)
wid = termf.winfo_id()
os.system('xterm -into %d -geometry 40x20 -sb &' % wid)

root.mainloop()
