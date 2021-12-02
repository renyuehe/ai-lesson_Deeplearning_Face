import numpy as np

def conv2d(img,kernel):
    h,w,c = img.shape
    kernel_h,kernel_w,kernel_inc,kernel_outc = kernel.shape
    out_h = h - kernel_h + 1
    out_w = w - kernel_w + 1
    feature_maps = np.zeros((out_h,out_w,kernel_outc))

    for oc in range(kernel_outc):
        for h in range(out_h):
            for w in range(out_w):
                for ic in range(kernel_inc):
                    patch = img[h:h+kernel_h,w:w+kernel_w,ic]
                    feature_maps[h,w,oc] += np.sum(patch * kernel[:,:,ic,oc])

    return feature_maps


if __name__ == '__main__':
    img = np.random.rand(3,3,3)
    kernel = np.random.rand(2,2,3,1)
    print(img)
    print("-----------")
    print(kernel)
    print("-----------")
    print(conv2d(img,kernel))
