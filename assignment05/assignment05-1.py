import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage import io, color
from skimage import exposure

def pprint(arr):
    print("type:{}".format(type(arr)))
    print("shape: {}, dimension: {}, dtype:{}".format(arr.shape, arr.ndim, arr.dtype))
    print("Array's Data:\n", arr)

file_image	= 'cau.jpg'
im_color 	= io.imread(file_image)
im_gray  	= color.rgb2gray(im_color)
#print('im_gray')
#pprint(im_gray)
pprint(im_gray)
kerX 		= np.array([[0,0,0],[1,-2,1],[0,0,0]])       #for x-derivative
kerY 		= np.array([[0,1,0],[0,-2,0],[0,1,0]])       #for y-derivative
blur        = np.array([[1,2,1],[2,4,2],[1,2,1]])/16     #for smoothing image
ker3 = kerX + kerY
im_conv		= signal.convolve2d(im_gray, ker3, boundary='symm', mode='same')
pprint(im_conv)
#print("im_conv")
#pprint(im_conv)
#p1 = plt.subplot(2,2,1)
#p1.set_title('color image')
#plt.imshow(im_color)
#plt.axis('off')
x = [1,2,3]
y = [1,2,3]
dx = [1,1,1]
dy = [1,1,1]

p2 = plt.figure()
plt.imshow(im_gray, cmap='gray')
plt.axis('off')

#p3 = plt.subplot(2,1,2)
##p3.set_title('convolution kernel')
#plt.imshow(ker3, cmap='gray')
#plt.axis('off')
def compute_magnitude(dx,dy):
    return (dx**2+dy**2)**0.5
def draw_arrow_plot(x,y,dx,dy):
    #data: dict with probabilities for the bases and pair transitions.
    #arrow_sep: separation arrows
    #display: 'Length','width','alpha' for arrow property
    #shape: 'full', 'left', 'right'
    #colors: 'r'

    #if(MinMag+MaxMag)/2 < mag    <MaxMag:
    #length = 10000
    #width = 100
    for i in rage(len(x)):
        plt.arrow(x[i], y[i], dx[i]*length,dy[i]*length,color = 'r',
                  width=width, head_width=1, head_length=2,
                  length_includes_head = True)
p4 = plt.figure()
plt.imshow(im_conv, cmap='gray')
plt.axis('off')
plt.arrow(100,100,1000,1000)
#plt.show()

#print(np.sum(im_conv))
#print(im_conv.item(50,100))    : 특정 좌표의 값 가져오기
print(im_conv.item(1320,1967))
