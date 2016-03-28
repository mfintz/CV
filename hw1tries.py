import numpy as np
import cv2
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
from PIL import Image


def selectfile():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


BLUE = [255,0,0]        # part 1
RED = [0,0,255]         # part 2
GREEN = [0,255,0]       # part 3
YELLOW = [0,255,255]    # part 4

DRAW_P1 = {'color' : BLUE, 'val' : 1}
DRAW_P2 = {'color' : RED, 'val' : 1}
DRAW_P3 = {'color' : GREEN, 'val' : 1}
DRAW_P4 = {'color' : YELLOW, 'val' : 1}

drawing = False         # flag for drawing curves
rect_over = True
thickness = 3           # brush thickness
value = DRAW_P1         # drawing initialized to P1

def onmouse(event,x,y,flags,param):
    global imgInput,img2,drawing,value,mask#,mask1,mask2,mask3,mask4#,ix,iy

    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(imgInput,(x,y),thickness,value['color'],-1)
        cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(imgInput,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(imgInput,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)


print(" Instructions: \n")
user_input1 = input('Enter parts number (2 - 4): ')
####### tut proverit' chtob ne bol'she 4
count = int(user_input1)


filename = selectfile()
imgInput = cv2.imread(filename)
#imgInput = cv2.imread('crem.jpg')

########### mozhet eto kak-to inache nado napisat'?
im=Image.open(filename)
#im=Image.open('crem.jpg')
(width,height) = im.size

img2 = imgInput.copy()                                # a copy of original image
mask1 = 2+np.zeros(imgInput.shape[:2],dtype = np.uint8) # mask initialized to P1
mask2 = 2+np.zeros(imgInput.shape[:2],dtype = np.uint8) # mask initialized to P2
mask3 = 2+np.zeros(imgInput.shape[:2],dtype = np.uint8) # mask initialized to P3
mask4 = 2+np.zeros(imgInput.shape[:2],dtype = np.uint8) # mask initialized to P4
output = np.zeros(imgInput.shape,np.uint8)            # output image to be shown
mask = mask1
cv2.imshow('',mask)
#input and output windows
cv2.namedWindow('output')
cv2.namedWindow('input')
cv2.setMouseCallback('input',onmouse)
#cv2.moveWindow('input',img.shape[1]+10,90)  #interesno, eto zachem?...
i = 1
print(" mark part 1 regions with left mouse button and after press 'n'\n")
cv2.imshow('output',output)
cv2.imshow('input',imgInput)

while(1):#for i in range(1, count):
    k = 0xFF & cv2.waitKey(1)
    cv2.imshow('output',output)
    cv2.imshow('input',imgInput)
    # key bindings
    if k == 27:    # esc to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('n'): # segment the image
        #cv2.imshow('output',output)
        #cv2.imshow('input',imgInput)
        i = i + 1
        ####### vot tut eshe dopisat' pro sluchaj bol'she count
        if i == 2:
            value = DRAW_P2
            mask = mask2
        elif i == 3:
            value = DRAW_P3
            mask = mask3
        elif i == 4:
            value = DRAW_P4
            mask = mask4
        print(" mark part %d regions with left mouse button and again press 'n'\n" % (i) )
        if i > count:
            cv2.destroyAllWindows()
            break
        ### a tut horosho by sdlat' reset ....

bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)
#rect = (0,0,width,height)

mask1, bgdModel, fgdModel = cv2.grabCut(img2,mask1,None,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
mask2, bgdModel, fgdModel = cv2.grabCut(img2,mask2,None,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
mask3, bgdModel, fgdModel = cv2.grabCut(img2,mask3,None,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
mask4, bgdModel, fgdModel = cv2.grabCut(img2,mask4,None,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
#mask11 = mask1*100
#plt.imshow(mask11),plt.colorbar(),plt.show()
#mask22 = mask2*150
#plt.imshow(mask22),plt.colorbar(),plt.show()
#mask33 = mask3*200
#plt.imshow(mask33),plt.colorbar(),plt.show()
#mask44 = mask4*250
#plt.imshow(mask44),plt.colorbar(),plt.show()

mask1 = np.where((mask1==2)|(mask1==0),0,1).astype('uint8')
mask2 = np.where((mask2==2)|(mask2==0),0,1).astype('uint8')
mask3 = np.where((mask3==2)|(mask3==0),0,1).astype('uint8')
mask4 = np.where((mask4==2)|(mask4==0),0,1).astype('uint8')

#mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
img_1 = img2*mask1[:,:,np.newaxis]
plt.imshow(img_1),plt.colorbar(),plt.show()
img_2 = img2*mask2[:,:,np.newaxis]
plt.imshow(img_2),plt.colorbar(),plt.show()
img_3 = img2*mask3[:,:,np.newaxis]
plt.imshow(img_3),plt.colorbar(),plt.show()
img_4 = img2*mask4[:,:,np.newaxis]
plt.imshow(img_4),plt.colorbar(),plt.show()

#output = cv2.bitwise_and(img2,img2, mask)#,mask=mask2)
#cv2.imshow('output',output)
#cv2.destroyAllWindows()



















#BLUE = [255,0,0]        # part 1
#RED = [0,0,255]         # part 2
#GREEN = [0,255,0]       # part 3
#BLACK = [0,0,0]         # part 4
#WHITE = [255,255,255]   # na vsyakij slucha

#DRAW_P1 = {'color' : BLUE, 'val' : 1}
#DRAW_PR_BG = {'color' : RED, 'val' : 2}
#DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
#DRAW_BG = {'color' : BLACK, 'val' : 4}
#DRAW_FG = {'color' : WHITE, 'val' : 5}


## setting up flags
#rect = (0,0,1,1)
#drawing = False         # flag for drawing curves
#rectangle = False       # flag for drawing rect
#rect_over = False       # flag to check if rect drawn
#rect_or_mask = 100      # flag for selecting rect or mask mode
#value = DRAW_FG         # drawing initialized to FG
#thickness = 3           # brush thickness

#def onmouse(event,x,y,flags,param):
#    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

#    # Draw Rectangle
#    if event == cv2.EVENT_RBUTTONDOWN:
#        rectangle = True
#        ix,iy = x,y

#    elif event == cv2.EVENT_MOUSEMOVE:
#        if rectangle == True:
#            img = img2.copy()
#            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
#            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
#            rect_or_mask = 0

#    elif event == cv2.EVENT_RBUTTONUP:
#        rectangle = False
#        rect_over = True
#        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
#        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
#        rect_or_mask = 0
#        print(" Now press the key 'n' a few times until no further change \n")

#    # draw touchup curves

#    if event == cv2.EVENT_LBUTTONDOWN:
#        if rect_over == False:
#            print("first draw rectangle \n")
#        else:
#            drawing = True
#            cv2.circle(img,(x,y),thickness,value['color'],-1)
#            cv2.circle(mask,(x,y),thickness,value['val'],-1)

#    elif event == cv2.EVENT_MOUSEMOVE:
#        if drawing == True:
#            cv2.circle(img,(x,y),thickness,value['color'],-1)
#            cv2.circle(mask,(x,y),thickness,value['val'],-1)

#    elif event == cv2.EVENT_LBUTTONUP:
#        if drawing == True:
#            drawing = False
#            cv2.circle(img,(x,y),thickness,value['color'],-1)
#            cv2.circle(mask,(x,y),thickness,value['val'],-1)

#if __name__ == '__main__':

#    # print documentation
#    print(__doc__)


#    filename = selectfile()

#    img = cv2.imread(filename)
#    img2 = img.copy()                               # a copy of original image
#    mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
#    output = np.zeros(img.shape,np.uint8)           # output image to be shown

#    # input and output windows
#    cv2.namedWindow('output')
#    cv2.namedWindow('input')
#    cv2.setMouseCallback('input',onmouse)
#    cv2.moveWindow('input',img.shape[1]+10,90)

#    print(" Instructions: \n")
#    #print(" Draw a rectangle around the object using right mouse button \n")

#    while(1):

#        cv2.imshow('output',output)
#        cv2.imshow('input',img)
#        k = 0xFF & cv2.waitKey(1)

#        # key bindings
#        if k == 27:         # esc to exit
#            break
#        elif k == ord('0'): # BG drawing
#            print(" mark background regions with left mouse button \n")
#            value = DRAW_BG
#        elif k == ord('1'): # FG drawing
#            print(" mark foreground regions with left mouse button \n")
#            value = DRAW_FG
#        elif k == ord('2'): # PR_BG drawing
#            value = DRAW_PR_BG
#        elif k == ord('3'): # PR_FG drawing
#            value = DRAW_PR_FG
#        elif k == ord('s'): # save image
#            bar = np.zeros((img.shape[0],5,3),np.uint8)
#            res = np.hstack((img2,bar,img,bar,output))
#            cv2.imwrite('grabcut_output.png',res)
#            print(" Result saved as image \n")
#        elif k == ord('r'): # reset everything
#            print("resetting \n")
#            rect = (0,0,1,1)
#            drawing = False
#            rectangle = False
#            rect_or_mask = 100
#            rect_over = False
#            value = DRAW_FG
#            img = img2.copy()
#            mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
#            output = np.zeros(img.shape,np.uint8)           # output image to be shown
#        elif k == ord('n'): # segment the image
#            print(""" For finer touchups, mark foreground and background after pressing keys 0-3
#            and again press 'n' \n""")
#            if (rect_or_mask == 0):         # grabcut with rect
#                bgdmodel = np.zeros((1,65),np.float64)
#                fgdmodel = np.zeros((1,65),np.float64)
#                cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
#                rect_or_mask = 1
#            elif rect_or_mask == 1:         # grabcut with mask
#                bgdmodel = np.zeros((1,65),np.float64)
#                fgdmodel = np.zeros((1,65),np.float64)
#            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)

#        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
#        output = cv2.bitwise_and(img2,img2,mask=mask2)

#    cv2.destroyAllWindows()
