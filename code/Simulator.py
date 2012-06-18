#!/usr/bin/python
# -*- coding: utf-8 -*- 

import wx

class Simulator(wx.App):
  
    def OnInit(self):
        frame = wx.Frame(None, title="Rough Fuzzy simulator")
        frame.Centre()
        frame.Show()
        self.SetTopWindow(frame)
        return True

if __name__ == '__main__':
  
    app = Simulator()
    app.MainLoop()