import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage import io, color
from skimage import exposure

# normalize the values of the input data to be [0, 1].
def normalize(data):
    data_normalized = (data - min(data)) / (max(data) - min(data))
    return(data_normalized)

#compute magnitude of vectors.
def compute_magnitude(dx,dy):
    #dx,dy are numpy arrays
    return (dx**2+dy**2)**0.5

#draw arrows which visualize the gradient.
#x,y,mean_x,mean_y are numpy 1 dimension arrays
def draw_arrows(x,y,mean_x,mean_y,sampling_size):
    #mean_mag is used for making arrows-length longer or shorter on same ratio.
    mean_mag = compute_magnitude(mean_x,mean_y)
    #find max magnitude. It is max length of arrow
    mag_max = mean_mag.max()
    for j in range(mean_mag.size):
        #multiply same ratio to all arrow length
        dx = mean_x[j]*((sampling_size/2)/mag_max)
        dy = mean_y[j]*((sampling_size/2)/mag_max)
        #Thesetting can be changed by sampling size.
        plt.arrow(x[j],y[j],dx,dy,color='b'
                ,width=sampling_size/50, head_width=sampling_size/10, head_length=sampling_size/10,length_includes_head = False)

#get the image name
def main_func(image_name):
    file_image	= image_name
    im_color 	= io.imread(file_image)
    im_gray  	= color.rgb2gray(im_color)

    #color image plot
    p1 = plt.figure()
    plt.title('color image')
    plt.imshow(im_color)
    plt.axis('off')

    #gray image plot
    p2 = plt.figure()
    plt.title('gray image')
    plt.imshow(im_gray, cmap='gray')
    plt.axis('off')
    plt.show()

    #kernels for computing x,y derivatives
    kerY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])             #for y-derivative
    kerX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])             #for x-derivative
    #convolved arrays
    im_kerX = signal.convolve2d(im_gray, kerX, boundary='symm', mode='same')
    im_kerY = signal.convolve2d(im_gray, kerY, boundary='symm', mode='same')

    #Derivatives plot
    p3 = plt.figure()
    plt.title('Derivative_x')
    plt.imshow(im_kerX, cmap='gray')
    plt.axis('off')

    p4 = plt.figure()
    plt.title('Derivative_y')
    plt.imshow(im_kerY, cmap='gray')
    plt.axis('off')

    plt.show()



    #this is the size of rows and column for sampling gradient.
    #If sampling size is 50, sampling square(50x50).
    sampling_size = 50

    list_mean_x = []
    list_mean_y = []
    list_x = []
    list_y = []

    deri_x = signal.convolve2d(im_gray, kerX, boundary='symm', mode='same')
    deri_y = signal.convolve2d(im_gray, kerY, boundary='symm', mode='same')
    #for gradient
    kerG = kerX + kerY
    im_conv		= signal.convolve2d(im_gray, kerG, boundary='symm', mode='same')


    #sampling squares and compute each mean in sliced arrays.
    for i in range(np.size(deri_y,0)//sampling_size):
        for k in range(np.size(deri_x,1)//sampling_size):
            x = k*sampling_size
            y = i*sampling_size
            next_x = (k+1)*sampling_size
            next_y = (i+1)*sampling_size
            x_mean = np.mean(deri_x[y:next_y,x:next_x])
            y_mean = np.mean(deri_y[y:next_y,x:next_x])
            list_mean_x.append(x_mean)
            list_mean_y.append(y_mean)
            list_x.append((x+next_x)/2)
            list_y.append((y+next_y)/2)

    #get numpy arrays of the results.
    mean_x = np.array(list_mean_x)
    mean_y = np.array(list_mean_y)
    x = np.array(list_x)
    y = np.array(list_y)


    p5 = plt.figure()
    plt.title('gradient image')
    plt.imshow(im_conv, cmap='gray')
    #draw arrows.
    draw_arrows(x,y,mean_x,mean_y,sampling_size)

    plt.axis('off')
    plt.show()

    #for smoothing image
    blur        = np.array([[0.5,1,0.5],[1,4,1],[0.5,1,0.5]])/8
    result_blur =  signal.convolve2d(im_gray, blur, boundary='symm', mode='same')

    own_kernel = np.array([[0,-1,-2],[1,1,-1],[0,1,2]])
    im_own = signal.convolve2d(im_gray, own_kernel, boundary='symm', mode='same')

    p6 = plt.figure()
    plt.title('smoothe image')
    plt.imshow(result_blur, cmap='gray')
    plt.axis('off')

    p7 = plt.figure()
    plt.title('own kernel convolution')
    plt.imshow(im_own, cmap='gray')
    plt.axis('off')

    plt.show()
main_func('example.jpg')
