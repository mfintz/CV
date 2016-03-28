from tkinter import *
from tkinter import filedialog
import cv2
from tkinter.filedialog import askopenfile
import numpy as np
from matplotlib import pyplot as plt




def my_func():
    fileName = askopenfile()
    print("File name is :" + fileName.name)
    image = plt.imread(fileName.name)    #if i want the image in graysacle ,0 should come after fileName.name #also , it used to be cv2.imread
   # cv2.imshow('image',image)
    plt.imshow(image)
    plt.colorbar()
    plt.show()


window=Tk()
window.title("Original")
window.geometry("500x500")
btn = Button(window, text="Open image to be segmented", command=my_func).pack()



window.mainloop()

