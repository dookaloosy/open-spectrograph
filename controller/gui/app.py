"""`controller gui` entry point (Qt imported here, lazily)."""

import sys


def run(port=None) -> int:
    from PySide6.QtWidgets import QApplication

    from controller.gui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    window = MainWindow(port)
    window.show()
    return app.exec()
