# -*- coding: utf-8 -*-
"""
Created on Sat Jan 1 18:43:18 2018

@author: MONSTER
"""

import sys
from PyQt4 import QtGui
from main import MainWindow


def main():
    app = QtGui.QApplication(sys.argv)

    mainWindow = MainWindow()
    mainWindow.show()

    return app.exec_()

if __name__ == "__main__":
    main()