import math
import os
import os.path
import sys
from PIL import Image

def image_rotate(in_file_path, out_dir):

    print(in_file_path)
    in_file = os.path.basename(in_file_path)
    file_name = ''.join(in_file.split('.')[:-1])

    img = Image.open(in_file_path)
    sx, sy = img.size
    cx, cy = sx/2.0, sy/2.0
    d = min(cx, cy)
    a = d*math.sqrt(2)
    area = (cx - a/2, cy - a/2, cx + a/2, cy + a/2)

    print('sx={0}, sy={1}'.format(sx,sy))
    print('cx={0}, cy={1}'.format(cx,cy))
    print('d={0:.3f}, a={1:.3f}'.format(d,a))

    #angle = -27
    #img_rot = img.rotate(angle)
    #img_rot.save("out.jpg")

    for angle in range(0, 360, 30):

        img_rot = img.rotate(angle)
        box = img_rot.crop(area)

        #box.show()
        out_file = out_dir + '/' \
            + '{0}_{1:03d}.jpg'.format(file_name, angle)
        box.save(out_file)


def rotate_all_dir(in_dir, out_dir):

    files = os.listdir(in_dir)
    
    for file_name in files:
        file_path = in_dir + '/' + file_name
        image_rotate(file_path, out_dir)


#-----------------------------------
if __name__ == '__main__':
    
    in_dir = '/ram/tmp/in'
    out_dir = '/ram/tmp/out'

    in_dir = in_dir.rstrip('/')
    out_dir = out_dir.rstrip('/')
    rotate_all_dir(in_dir, out_dir)