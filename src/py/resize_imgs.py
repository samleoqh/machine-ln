# Image pre-process tool
# This scrip should be put in the same folder of images
# after run, resized impages will be saved into a new created folder
# /resized/..image.jpg
# change size prarameter according to your needs, the default settings are
# 512x512
from PIL import Image
import os, sys

# for python3, needed define cmp function,
# for python2.7, don't need define it, can remove it.
def cmp(a,b):
    return (a > b) - (a < b)

def resizeImage(infile, output_dir="resized/", size=(512,512)):
     outfile = os.path.splitext(infile)[0]+"_resized"
     extension = os.path.splitext(infile)[1]

     if (cmp(extension, ".jpg")):
        return

     if infile != outfile:
        try :
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(output_dir+outfile+extension,"JPEG")
        except IOError:
            print ("cannot reduce image for {}".format(infile))


if __name__=="__main__":
    output_dir = "resized"
    dir = os.getcwd()

    if not os.path.exists(os.path.join(dir,output_dir)):
        os.mkdir(output_dir)

    for file in os.listdir(dir):
        resizeImage(file)