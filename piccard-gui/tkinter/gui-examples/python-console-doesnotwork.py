import wx
import wx.stc as stc
import code
import sys
import __main__

#x = code.InteractiveConsole() #x.interact("A kind of Python interpreter")
class II(code.InteractiveInterpreter):
    def __init__(self, locals):
        code.InteractiveInterpreter.__init__(self, locals)

    def Runit(self, cmd):
        code.InteractiveInterpreter.runsource(self, cmd)

class PySTC(stc.StyledTextCtrl):
    def __init__(self, parent, ID, pos=(10,10), size=(700, 600), style=0):
        stc.StyledTextCtrl.__init__(self, parent, ID, pos, size, style)

        sys.stdout = self
        sys.stderr = self
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyPressed)
        self.cmd = ''
        self.lastpos = self.GetCurrentPos()

    def SetInter(self, interpreter):
        self.inter = interpreter
        
    def write(self, ln):
        self.AppendTextUTF8('%s'%str(ln))
        self.GotoLine(self.GetLineCount())

    def OnKeyPressed(self, event):
        self.changed = True     # Records what's been typed in
        char = event.GetKeyCode()   # get code of keypress
        if (self.GetCurrentPos() < self.lastpos) and (char <314) or (char > 317):
            pass
            # need to check for arrow keys in this
        elif char == 13:
            """
            What to do if <enter> is pressed? It depends if there
            are enough instructions
            """
            lnno = self.GetCurrentLine()
            ln = self.GetLine(lnno)
            self.cmd = self.cmd + ln + '\r\n'
            self.NewLine()
            self.tabs = ln.count('\t') #9
            if (ln.strip() == '') or ((self.tabs < 1) and (':' not in ln)):
                # record command in command list
                self.cmd = self.cmd.replace('\r\n','\n')
                # run command now
                self.inter.Runit(self.cmd)
                self.cmd = ''
                self.lastpos = self.GetCurrentPos()
            else:
                if ':' in ln:
                    self.tabs = self.tabs + 1
                self.AppendText('\t' * self.tabs)
                # change cursor position now
                p = self.GetLineIndentPosition(lnno + 1)
                self.GotoPos(p)
        else:
            event.Skip() # ensure keypress is shown

class ecpintframe(wx.Frame):
    def __init__(self, *args, **kwds):
        kwds["size"] = (700,600)
        wx.Frame.__init__(self, *args, **kwds)
        self.ed = PySTC(self, -1)

if __name__ == '__main__':
    Ecpint = wx.PySimpleApp(0)
    I = II(None)
    win = ecpintframe(None, -1, "EcPint - Interactive intepreter")
    win.Show()
    win.ed.SetInter(I)
    Ecpint.MainLoop()
