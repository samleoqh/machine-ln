# image datasets enxtend by image data generator
# should genereate one type/class by one type/class
# the folder structure as blow
# note only one type in the train folder for one time.
# the code need reflector in future

"""directory structure:
```
dataset/
    train/
        Type_1/
            001.jpg
            002.jpg
            ...
"""

from keras.preprocessing.image import ImageDataGenerator

img_dir = '/Users/liuqh/Desktop/dataset/train'
sav_dir = '/Users/liuqh/Desktop/new'

datagen = ImageDataGenerator(
    rotation_range = 90,
    #width_shift_range = 0.2,
    #height_shift_range = 0.2,
    #zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)


i = 1
for batch in datagen.flow_from_directory(img_dir,
                                         target_size=(224,224),
                                         shuffle=False,
                                         batch_size= 100,
                                         save_prefix='_gen',
                                         save_to_dir=sav_dir):
    i += 1
    if i > 66:
        break


