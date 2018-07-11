import math
from PIL import Image

img = Image.open("in.jpg")
angle = -27
img_rot = img.rotate(angle)
#img_rot.save("out.jpg")

sx, sy = img.size
cx, cy = sx/2.0, sy/2.0
d = min(cx, cy)
a = d*math.sqrt(2)

print('sx={0}, sy={1}'.format(sx,sy))
print('cx={0}, cy={1}'.format(cx,cy))
print('d={0}, a={1}'.format(d,a))

area = (cx - a/2, cy - a/2, cx + a/2, cy + a/2)
box = img_rot.crop(area)
#box.show()
box.save('out_box.jpg')