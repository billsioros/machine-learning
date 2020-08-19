
import numpy as np
from PIL import Image

if __name__ == "__main__":
    image = Image.open('./img/F1.jpg')
    image = image.resize((100, 100))

    print(np.asarray(image).shape)
    print(np.array(image.getdata()).shape)
    print(np.array(image.getdata()))