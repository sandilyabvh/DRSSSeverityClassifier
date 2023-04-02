import os
import tkinter as tk
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")
        self.master.geometry("800x600")

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        self.image_path_label = tk.Label(self.master, text="")
        self.image_path_label.pack()

        self.current_index = 0
        file_paths = '/storage/home/hpaceice1/vbalemarthy3/fml_project/DRSSSeverityClassifier/image_file_paths.txt'
        with open(file_paths, 'r') as f:
            self.image_file_paths = f.readlines()

        self.show_image()

        self.next_button = tk.Button(self.master, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.previous_button = tk.Button(self.master, text="Previous", command=self.previous_image)
        self.previous_button.pack(side=tk.LEFT)

    def show_image(self):
        self.image_path = self.image_file_paths[self.current_index].strip()

        self.image = Image.open(self.image_path)
        self.resized_image = self.image.resize((600, 400))
        self.photo = ImageTk.PhotoImage(self.resized_image)
        self.image_label.config(image=self.photo)

        self.image_path_label.config(text=self.image_path)

    def next_image(self):
        if self.current_index < len(self.image_file_paths) - 1:
            self.current_index += 1
            self.show_image()

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

if __name__ == '__main__':
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
