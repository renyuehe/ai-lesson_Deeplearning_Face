import numpy as np


def init_centroids(k, n_features):
    return np.random.random(k * n_features).reshape((k, n_features))


def update_centroids(points, centroid_index):
    k = max(centroid_index)+1
    new_centroids = np.zeros((10,2))
    for i in range(k):
        new_centroids[i]=points[centroid_index==i].mean(axis=0)
    return new_centroids


def distance(pointA, pointB):
    return np.sqrt((pointA[0]-pointB[0])**2+(pointA[1]-pointB[1])**2)


def belongs2(point, centroids):
    index = 0
    min_distance = np.inf
    for i in range(len(centroids)):
        d = distance(point, centroids[i])
        if d<min_distance:
            min_distance=d
            index=i
    return index


def update_index(points, centroids):
    n_samples = len(points)
    new_indeces = np.zeros((n_samples))
    for i, point in enumerate(points):
        new_indeces[i] = belongs2(point, centroids)
    new_indeces = new_indeces.astype(int)
    return new_indeces


def my_kmeans(points):
    centroids = init_centroids(10, 2) #十个中心点
    indeces=update_index(points, centroids)
    old_indeces = indeces
    for i in range(1000):
        centroids=update_centroids(points, indeces)
        indeces=update_index(points, centroids)
        if np.array_equal(indeces, old_indeces):
            print('converge', i)
            break
        else:
            old_indeces=indeces
    return centroids, indeces


if __name__ == '__main__':
    points = np.array([[1,1],[2,2],[3,3],[1,2],[2,1],[2,3],[3,2],[5,5],[6,6],[9,9],[11,11],[10,11],[11,10],[20,20],[30,30]])
    print(my_kmeans(points))
