from PIL import Image 
import numpy as np

image = Image.open('data/accuracy/bubbles.png') 
image_array = np.array(image)
print(image_array[:,:,0].mean())

image.show() 
