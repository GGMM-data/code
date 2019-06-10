from PIL import Image

image = Image.open('/home/mxxmhh/Downloads/DSC_0748.jpg')

w,h = image.size

print(w,h)

image.thumbnail((w//2,h//2))
print("resize image to: %sx%s" % (w//2,h//2))

image.save("mxx.pdf","pdf")


image2 = Image.new('RGB',(300,300),(255,255,255))#mode, size( 2-tuple, width,height),color

image2.show()
