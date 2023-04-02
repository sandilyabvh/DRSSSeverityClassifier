import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

class ImageViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")
        self.master.geometry("800x600")

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        self.directory_button = tk.Button(self.master, text="Select directory", command=self.select_directory)
        self.directory_button.pack()

        self.image_name_label = tk.Label(self.master, text="")
        self.image_name_label.pack()

        self.previous_button = tk.Button(self.master, text="Previous image", command=self.previous_image)
        self.previous_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(self.master, text="Next image", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=10)

    def select_directory(self):
        self.directory_path = filedialog.askdirectory(title="Select directory")

        self.images = []
        self.image_names = []
        for file_name in os.listdir(self.directory_path):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                self.images.append(Image.open(os.path.join(self.directory_path, file_name)))
                self.image_names.append(file_name)

        self.current_index = 0
        self.show_image()

    def show_image(self):
        self.image = self.images[self.current_index]
        self.image_name = self.image_names[self.current_index]
        self.image_name_label.config(text=self.image_name)

        self.resized_image = self.image.resize((600, 400))
        self.photo = ImageTk.PhotoImage(self.resized_image)
        self.image_label.config(image=self.photo)

    def next_image(self):
        if self.current_index < len(self.images) - 1:
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
