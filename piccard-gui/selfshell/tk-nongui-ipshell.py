from IPython.frontend.terminal.embed import InteractiveShellEmbed
# create ipshell *before* calling enable_gui
# it is important that you use instance(), instead of the class
# constructor, so that it creates the global InteractiveShell singleton
ipshell = InteractiveShellEmbed.instance()

import IPython.lib.inputhook
IPython.lib.inputhook.enable_gui(gui='tk')

def foo():
    # without inputhook, 'a' is found just fine
    exec 'a=123' in globals()
    # all calls to instance() will always return the same object
    ipshell = InteractiveShellEmbed.instance()
    ipshell()

foo()
