from ui import SplitterUI
import tkinter.messagebox as messagebox

if __name__ == "__main__":
    try:
        app = SplitterUI()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Error", str(e))
