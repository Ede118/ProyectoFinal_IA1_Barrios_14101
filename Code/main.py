# Code/main.py

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6 import QtWidgets

from Code.ui.main_window import MainWindow
from Code.app.AppController import AppController


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)

    controller = AppController()  # usa ImgOrchestrator, AudioOrchestrator, BayesAgent
    win = MainWindow(controller=controller)

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
