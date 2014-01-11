"""Quick code snippets for embedding IPython into other programs.

See example-embed.py for full details, this file has the bare minimum code for
cut and paste use once you understand how to use the system."""

#---------------------------------------------------------------------------
# This code loads IPython but modifies a few things if it detects it's running
# embedded in another IPython session (helps avoid confusion)

try:
    get_ipython
except NameError:
    banner=exit_msg=''
else:
    banner = '*** Nested interpreter ***'
    exit_msg = '*** Back in main IPython ***'

# First import the embed function
from IPython.terminal.embed import InteractiveShellEmbed
# Now create the IPython shell instance. Put ipshell() anywhere in your code
# where you want it to open.
ipshell = InteractiveShellEmbed(banner1=banner, exit_msg=exit_msg)
                            
#---------------------------------------------------------------------------
# This code will load an embeddable IPython shell always with no changes for
# nested embededings.

from IPython import embed
# Now embed() will open IPython anywhere in the code.

#---------------------------------------------------------------------------
# This code loads an embeddable shell only if NOT running inside
# IPython. Inside IPython, the embeddable shell variable ipshell is just a
# dummy function.

try:
    get_ipython
except NameError:
    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed()
    # Now ipshell() will open IPython anywhere in the code
    # What does this do? ipshell = InteractiveShellEmbed.instance()

    # Why would I want an inputhook?
    # import IPython.lib.inputhook
    # IPython.lib.inputhook.enable_gui(gui='tk')

else:
    # Define a dummy ipshell() so the same code doesn't crash inside an
    # interactive IPython
    def ipshell(): pass

#******************* End of file <example-embed-short.py> ********************

#ipshell()

#ipshell.ex('print "10 * 20 = ", 10*20')
print """

Have some fun with the IPython exec ipshell

"""

cell = """
a = 'Hallo'
b = 20
print "a, b = ", a, b
"""
command = compile(cell, '<string>', 'exec')
ipshell.ex(command)
cell = """
print "a, b = ", a, b
"""
command = compile(cell, '<string>', 'exec')
ipshell.ex(command)
print """

Ok, done. Now see if we can do a full cell with 'run_cell'

"""

cell = """
import numpy as np
x = np.arange(0, 3, 0.1)
y = np.sin(x)
print y
print a, b
"""
ipshell.run_cell(cell)

print """

Yep. All cool. Now see if we can re-direct the output somewhere...

"""
print "See other example"

print """

Can we push and pull variables/object into the interactive shell?

"""
ipshell.run_cell("Rutger = None ; Sonja = 'zijn broer'")
Rutger = 'een Nederlander'
Sonja = 'zijn zus'
cell = """
if Rutger is None:
    print "Don't have Rutger"
else:
    print "Rutger = ", Rutger
print "Sonja = ", Sonja
"""
ipshell.run_cell(cell)
ipshell.push({'Rutger':Rutger}, interactive=True)
ipshell.run_cell(cell)
#ipshell.pull({'Sonja':Sonja)
print "Now, locally, we have Sonja = ", Sonja
print "Embedded, we have Sonja = ", ipshell.ns_table['user_local']['Sonja']
