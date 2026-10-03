"""
Punto de entrada del sistema de reconocimiento facial.

Ejecutar con:
    python main.py
"""

import tkinter as tk

from gui_app import FacialRecognitionApp


def main():
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
