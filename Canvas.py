import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
import matplotlib.pyplot as plt

# class mainWindow():
#     def __init__(self):
#         self.root = tk.Tk()
#         self.frame = tk.Frame(self.root, width=100*5, height=100*5)
#         self.frame.pack()
#         self.canvas = tk.Canvas(self.frame, width=100*5, height=100*5)
#         self.canvas.place(x=-2, y=-2)
#
#     def start(self, cishu):
#             data=np.array(np.random.randn(100, 100, 3), dtype=int)
#             self.im=Image.fromarray(data, 'RGB')
#             self.photo = ImageTk.PhotoImage(image=self.im)
#             self.canvas.create_image(20, 20, image=self.photo, anchor=tk.NW)
#             self.root.update()
#             self.root.mainloop()
#
# a = mainWindow()
# a.start(3)
'''
UNIT = 5
MAZE_H = 100
MAZE_W = 100

class show(tk.Tk, object):
    def __init__(self):
        super(show, self).__init__()
        self.title('traffic')
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.observation = np.random.randn(100, 100, 3)
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, height=500, width=500)
        tmp = Image.fromarray(np.array(self.observation), 'RGB').resize((500, 500))
        img = ImageTk.PhotoImage(tmp)
        self.image = self.canvas.create_image(20, 20, image=img)
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.image)
        tmp = Image.fromarray(np.array(self.observation), 'RGB').resize((500, 500))
        tmp.show()
        self.image = self.canvas.create_image(20, 20, image=ImageTk.PhotoImage(tmp))
        self.update()

    def step(self, observation):
        print("test")
        plt.close()
        # self.update()
        self.observation = observation
        # self.canvas.delete(self.image)
        tmp = Image.fromarray(np.array(self.observation), 'RGB').resize((500, 500))
        # self.image = self.canvas.create_image(20, 20, image=ImageTk.PhotoImage(tmp))
        # self.update()
        plt.imshow(tmp)
        plt.show()
        print("test")

a = show()

if __name__ == '__main__':
    # a.reset()
    for i in range(10):
        a.step(np.zeros([100, 100, 3]))
'''
def test(times):
    root = tk.Tk()
    canvas = tk.Canvas(root, height=500, width=500)
    canvas.pack()
    tmp = np.random.randn(100, 100, 3)
    img = ImageTk.PhotoImage(Image.fromarray(tmp, "RGB").resize((800, 800)))
    canv_pic = canvas.create_image(20, 20, image=img)
    root.update()
    for i in range(times):
        canvas.delete(canv_pic)
        tmp = np.random.randn(100, 100, 3)
        img = ImageTk.PhotoImage(Image.fromarray(tmp, "RGB").resize((800, 800)))
        canv_pic = canvas.create_image(20, 20, image=img)
        root.update()
        print("finish one test")

test(10)
