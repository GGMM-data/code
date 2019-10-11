import tensorboardX
import math

if __name__ == "__main__":
    writer = tensorboardX.SummaryWriter()
    funcs = {"sin": math.sin, "cos": math.cos}
    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            value = fun(angle_rad)
            writer.add_scalar(name, value, angle)
    writer.close()
