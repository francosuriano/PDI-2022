from tkinter.ttk import Sizegrip
import imageio
import numpy as np
import matplotlib.pyplot as plt

MAT_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                        [0.596,-0.275,-0.321],
                        [0.211,-0.523, 0.311]])

def apply_matrix(img, M):
    return np.matmul(img.reshape((-1,3)), M.T).reshape(img.shape)
def rgb2yiq(img):
    return apply_matrix(img, MAT_RGB2YIQ)
def yiq2rgb(img):
    return apply_matrix(img, np.linalg.inv(MAT_RGB2YIQ))
def plot_hist(im, bins, ax, cumulative=False):
    counts, borders = np.histogram(im if im.ndim==2 else rgb2yiq(im)[...,0], bins=bins, range=(0,1))
    ax.bar(range(len(counts)), np.cumsum(counts) if cumulative else counts)
    plt.xticks(ax.get_xticks(), labels=np.round(ax.get_xticks()/bins,2))
    plt.grid(alpha=0.3)
    return counts,bins


#-------------------------------------------------------------------------------------------------------#
        
def Norm01(img):
    if img.shape!=2:
        img= rgb2yiq(img)[:,:,0]
        
    img_nueva = img
    maximo = np.amax(img)
    minimo = np.amin(img)
    img_nueva[np.where(img_nueva==maximo)]=1
    img_nueva[np.where(img_nueva==minimo)]=0
    
    return img_nueva

"""
img_rgb = imageio.imread('imageio:coffee.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(Norm01(img_rgb), 'gray', vmin=0, vmax=1)
plot_hist(Norm01(img_rgb), 50, axes[1])

img_rgb = imageio.imread('imageio:coffee.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_rgb, 'gray', vmin=0, vmax=1)
plot_hist(img_rgb, 50, axes[1])

"""

#-------------------------------------------------------------------------------------------------------------------#

def Norm_Percentil(img,p=1):
    if img.shape!=2:
        img= rgb2yiq(img)[:,:,0]
        
    img_nueva = img
    # Prueba todo a cero
    #img_nueva[:] = 0
   
    img_nueva[np.where(img_nueva<=np.percentile(img_nueva,p))] = 0
    img_nueva[np.where(img_nueva>=np.percentile(img_nueva,100-p))] = 1
    
    return img_nueva

"""
img_rgb = imageio.imread('imageio:coffee.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(Norm_Percentil(img_rgb,10), 'gray', vmin=0, vmax=1)
plot_hist(Norm_Percentil(img_rgb,10), 50, axes[1])

img_rgb = imageio.imread('imageio:coffee.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_rgb, 'gray', vmin=0, vmax=1)
plot_hist(img_rgb, 50, axes[1])

plt.show()

"""
#-------------------------------------------------------------------------------------------------------------------#

def Gamma(img,p=1,alfa=0):
    if img.shape!=2:
        img= rgb2yiq(img)[:,:,0]
        
    img_nueva = img

    img_nueva[np.where(img_nueva==np.percentile(img_nueva,p))] = 0
    img_nueva[np.where(img_nueva==np.percentile(img_nueva,100-p))] = 1
    
    #correción gamma

    # alfa > 0 -> aumenta la luminosidad
    # alfa < 0 -> disminuye la luminosidad
    # vout = vin^gamma

    img_nueva = np.clip((img_nueva**(2**(-alfa))),0,1)

    return img_nueva

"""
img_rgb = imageio.imread('imageio:coffee.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(Gamma(img_rgb,1,2), 'gray', vmin=0, vmax=1)
plot_hist(Gamma(img_rgb,1,2), 50, axes[1])

img_rgb = imageio.imread('imageio:coffee.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_rgb, 'gray', vmin=0, vmax=1)
plot_hist(img_rgb, 50, axes[1])
"""

#--------------------------------------------------------------------------------------------------------------------#

def Lineal(img,p=1,alfa=0,x=[],y=[]):
    
    if (np.prod(x)==np.prod(y)and (np.amin(x)>=0)and (np.amax(x)<=1) and (np.amin(y)>=0) and (np.amax(y)<=1)):

        if img.shape!=2:
            img= rgb2yiq(img)[:,:,0]
        
        #correción percentil
        img_nueva = img
        img_nueva[np.where(img_nueva==np.percentile(img_nueva,p))] = 0
        img_nueva[np.where(img_nueva==np.percentile(img_nueva,100-p))] = 1
    
        #correción gamma
        # alfa > 0 -> aumenta la luminosidad
        # alfa < 0 -> disminuye la luminosidad
        img_nueva = np.clip((2**(-alfa))*img_nueva,0,1)
        
        #correción lineal
        img_nueva = img
        img_nueva = np.interp(img_nueva,x,y)
        return img_nueva

    else:
        print('Ingrese dos arreglos del mismo tamaño y con rango : [0,1]')
        return 0



x = np.array([0, 0.2,  0.8,  1])
y = np.array([0, 0.05, 0.95, 1])
p = 1
alfa = 0

img_rgb = imageio.imread('imageio:chelsea.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(Lineal(img_rgb,p,alfa,x,y), 'gray', vmin=0, vmax=1)
plot_hist(Lineal(img_rgb,p,alfa,x,y), 25, axes[1])


"""
img_rgb = imageio.imread('imageio:coffee.png')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_rgb, 'gray', vmin=0, vmax=1)
plot_hist(img_rgb, 50, axes[1])
"""


#------------------------------------------------------------------------------------------------------------------#

# Uniformalización

# Cargado de datos

img_rgb = imageio.imread('cat 1 (1).jpg')/255
img_gray = rgb2yiq(img_rgb)[:,:,0]
img_x = img_gray.copy()
rows= img_x.shape[0]
cols= img_x.shape[1]


# calcula en que bin está un pixel dada la cantidad de bins y el pixel
def CalcularBin(bins, pixel):
    w = 1/bins
    #print(np.trunc(pixel/w))
    return (np.trunc(pixel/w))


# calcula el percentil del pixel dado el vector counts, size (numero total de pixeles), los bins y el pixel
def CalcularPerc(counts,size,bins,pixel):
    aux = 0
    bin = CalcularBin(bins,pixel)

    for i in range (0,int(bin)-1):
        aux = aux + counts[i]   

    aux = aux/size
    #print(aux)
    return aux

img_x = Norm01(img_x)

_, axes = plt.subplots(1, 2, figsize=(15,5))
counts,bins = plot_hist(img_x, 50,axes[0])

#print(img_x[40,40])
#CalcularPerc(counts,rows*cols,50,img_x[40,40])


# Para cada pixel se calcula en que percentil está y se le asigna dicho valor

def Uni(img_x,counts,rows,cols,bins,img_gray):

    #para cada pixel, se calcula el percentil en el que está y se le asigna ese valor
    for i in range(rows):
        for j in range(cols):
            img_x[i,j] = CalcularPerc(counts,rows*cols,bins,img_gray[i,j])

    return img_x


# Prueba con imágen

img_x=Uni(img_x,counts,rows,cols,bins,img_gray)
counts,bins = plot_hist(img_x, 50, axes[1])
_, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(img_gray,'gray')
axes[1].imshow(img_x,'gray')
plt.show()




