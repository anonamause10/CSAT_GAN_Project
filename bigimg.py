import sys
from PIL import Image

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("lores-pokemon/pokemon-lowres/") if isfile(join("lores-pokemon/pokemon-lowres/", f))]
total_width = 96*30
max_height = 96*25

new_im = Image.new('RGB', (total_width, max_height))


i = 0
for x in range(30):
    for y in range(25):
        new_im.paste(Image.open("lores-pokemon/pokemon-lowres/"+onlyfiles[i]),(x*96,y*96))
        i+=1


new_im.save('test.jpg')