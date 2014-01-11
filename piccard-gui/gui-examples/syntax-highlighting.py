from Tkinter import *
import keyword
from string import ascii_letters, digits, punctuation, join

class SyntaxHighlightingText(Text):

    tags = {'kw': 'orange',
            'int': 'red'}

    def __init__(self, root):
        Text.__init__(self, root)
        self.config_tags()
        self.characters = ascii_letters + digits + punctuation

        self.bind('<Key>', self.key_press)

    def config_tags(self):
        for tag, val in self.tags.items():
            self.tag_config(tag, foreground=val)

    def remove_tags(self, start, end):
        for tag in self.tags.keys():
            self.tag_remove(tag, start, end)

    def key_press(self, key):
        cline = self.index(INSERT).split('.')[0]
        lastcol = 0
        char = self.get('%s.%d'%(cline, lastcol))
        while char != '\n':
            lastcol += 1
            char = self.get('%s.%d'%(cline, lastcol))

        buffer = self.get('%s.%d'%(cline,0),'%s.%d'%(cline,lastcol))
        tokenized = buffer.split(' ')

        self.remove_tags('%s.%d'%(cline, 0), '%s.%d'%(cline, lastcol))

        start, end = 0, 0
        for token in tokenized:
            end = start + len(token)
            if token in keyword.kwlist:
                self.tag_add('kw', '%s.%d'%(cline, start), '%s.%d'%(cline, end))
            else:
                for index in range(len(token)):
                    try:
                        int(token[index])
                    except ValueError:
                        pass
                    else:
                        self.tag_add('int', '%s.%d'%(cline, start+index))

            start += len(token)+1

if __name__ == '__main__':
    root = Tk()
    sht = SyntaxHighlightingText(root)
    sht.pack()
    root.mainloop()
