import math
from PIL import Image

in_file = "f01.jpg"
file_name = ''.join(in_file.split('.')[:-1])

img = Image.open(in_file)

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
    box.save('{0}_{1:03d}.jpg'.format(file_name, angle))
