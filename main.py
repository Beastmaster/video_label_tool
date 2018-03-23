'''

'''

import sys
from PyQt5.QtWidgets import QApplication
from ui_main import *



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    ui.read_config()
    sys.exit(app.exec_())


