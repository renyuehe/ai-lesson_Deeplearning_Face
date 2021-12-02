import random

xs = [x for x in range(100)]
ys = [x*3+2+random.random()/10 for x in xs]

lr = 0.0001

def GD():
    w = random.random()
    b = random.random()
    for i in range(1000000):
        for x, y in zip(xs, ys):
            h = w * x + b
            o = h - y
            loss = o ** 2

            dw = 2 * o * x
            db = 2 * o

            w = w - dw * lr
            b = b - db * lr

            if i % 100000 == 0:
                print(w,b,loss)

if __name__ == '__main__':
    GD()