from tkinter import *
from tkinter import filedialog
import cv2
from tkinter.filedialog import askopenfile


def my_func():
    #print("hello world")
    #fileName = askopenfile()
    #print(fileName.name)
    #file=fileName.name
    image = cv2.imread('Capture.jpg',0)
    cv2.imshow('image',image)

window = Tk()
window.title("something")
window.geometry("500x500")
btn = Button(window, text="im a button", command=my_func).pack()



window.mainloop()

