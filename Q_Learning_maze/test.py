import numpy as np
import sys
if sys.version_info.major == 2:
	import Tkinter as tk
else:
	import tkinter as tk
from PIL import Image, ImageTk

# import Tkinter  # 导入Tkinter模块
from PIL import Image, ImageTk

root = tk.Tk()
canvas = tk.Canvas(root,
						width=500,  # 指定Canvas组件的宽度
						height=600,  # 指定Canvas组件的高度
						bg='white')  # 指定Canvas组件的背景色
# im = Tkinter.PhotoImage(file='img.gif')     # 使用PhotoImage打开图片
image = Image.open("image_speed.jpeg")
im = ImageTk.PhotoImage(image)

canvas.create_image(300, 300, image=im)  # 使用create_image将图片添加到Canvas组件中
canvas.create_text(213, 445,  # 使用create_text方法在坐标（302，77）处绘制文字
				   text='Use Canvas'  # 所绘制文字的内容
				   , fill='gray')  # 所绘制文字的颜色为灰色
# create grids
for c in range(0, 400, 40):
	x0, y0, x1, y1 = c, 0, c, 400
	canvas.create_line(x0, y0, x1, y1)
for r in range(0, 400, 40):
	x0, y0, x1, y1 = 0, r, 400, r
	canvas.create_line(x0, y0, x1, y1)
canvas.create_text(300, 75,
				   text='Use Canvas',
				   fill='blue')
canvas.pack()  # 将Canvas添加到主窗口

root.update()

root.mainloop()

