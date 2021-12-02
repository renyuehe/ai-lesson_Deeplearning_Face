import numpy as np

def cal_confuse_matrix(y_pred,y_true,n_classes):
    matrix = np.zeros(shape=(n_classes,n_classes))
    for p,t in zip(y_pred[0],y_true[0]):
        matrix[t-1][p-1] += 1
    return matrix

if __name__ == '__main__':
    y_pred = np.array([[1,1,1,2,2,2,3,3,3,1]])
    y_true = np.array([[1,2,3,1,2,3,1,2,3,2]])
    matrix = cal_confuse_matrix(y_pred,y_true,n_classes=3)
    print(matrix)