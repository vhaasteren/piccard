
from Tkinter import * 
import ttk 

root = Tk() 

nb = ttk.Notebook(root) 
nb.pack(fill='both', expand=1) 
t = Text(nb) 
nb.add(t, text='foo') 
c = Canvas(nb) 
nb.add(c, text='bar') 

def on_button_3(event): 
    if event.widget.identify(event.x, event.y) == 'label': 
        index = event.widget.index('@%d,%d' % (event.x, event.y)) 
        print event.widget.tab(index, 'text') 

nb.bind('<3>', on_button_3) 
root.mainloop() 
