import numpy as np


# def bn_forward_naive(x, gamma, beta, running_mean, running_var, mode = "trian", eps = 1e-5, momentum = 0.9):
# 	n, ic, ih, iw = x.shape
# 	out = np.zeros(x.shape)
# 	if mode == 'train':
# 		batch_mean = np.zeros(running_mean.shape)
# 		batch_var = np.zeros(running_var.shape)
# 		for i in range(ic):
# 			batch_mean[i] = np.mean(x[:, i, :, :])
# 			batch_var[i] = np.sum((x[:, i, :, :] - batch_mean[i]) ** 2 ) / (n * ih * iw)
# 		for i in range(ic):
# 			out[:, i, :, :] = (x[:, i, :, :] - batch_mean[i]) / np.sqrt(batch_var[i] + eps)
# 			out[:, i, :, :] = out[:, i, :, :] * gamma[i] + beta[i]
# 		#update
# 		running_mean = running_mean * momentum + batch_mean * (1 - momentum)
# 		running_var = running_var * momentum + batch_var * (1 - momentum)
# 	elif mode == 'test':
# 		for i in range(ic):
# 			out[:, i, :, :] = (x[:, i, :, :] - running_mean[i]) / np.sqrt(running_var[i] + eps)
# 			out[:, i, :, :] = out[:, i, :, :] * gamma[i] + beta[i]
# 	else:
# 		raise ValueError('Invalid forward BN mode: %s' % mode)
# 	return out


def BN(x,gamma,beta,running_mean,running_var,mode="train",momentum = 0.9,eps = 1e-5):
    n,c,h,w = x.shape
    out = np.zeros(shape=x.shape)
    if mode == "train":
        batch_mean = np.zeros(running_mean.shape)
        batch_var = np.zeros(running_var.shape)
        for i in range(c):
            batch_mean[i] = np.mean(x[:,i,:,:])
            batch_var[i] = np.sum((x[:,i,:,:]-batch_mean)**2) / n*h*w
        for i in range(c):
            out[:,i,:,:] = (x[:,i,:,:]-batch_mean[i]) / np.sqrt(batch_var[i] + eps)
            out[:,i,:,:] = out[:,i,:,:] * gamma[i] + beta[i]

        running_mean = running_mean * momentum + batch_mean*(1-momentum)
        running_var = running_var *momentum + batch_var*(1-momentum)

    elif mode == "test":
        for i in range(c):
            out[:,i,:,:] = (x[:,i,:,:] - running_mean[i]) / np.sqrt(running_var[i] + eps)
            out[:,i,:,:] = out[:,i,:,:] * gamma[i] + beta[i]

    else:
        raise ValueError("Invalid foward BN mode: %s" % mode)

