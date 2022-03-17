import cv2
import numpy as np

def calc_distance(x1, x2):
    start = True
    for i in range(len(x2)):
        dist = np.sqrt(np.sum(np.square(x1 - x2[i,:]), axis = 1)).reshape((-1,1))
        if start == True:
            euclidean_dist = dist
            start = False
        else:
            euclidean_dist = np.hstack((euclidean_dist, dist))
    
    return euclidean_dist

def assign_clusters(center, pixels):
    cluster = np.zeros(pixels.shape[0])
    e_dist = calc_distance(pixels, center)
    cluster = np.argmin(e_dist, axis = 1)

    return cluster

def update_centers(pixels, clusters ,centers, n_cluster ):
    updated_center = np.zeros(centers.shape)
    for i in range(n_cluster):
        updated_center[i] = np.mean(pixels[clusters == i], axis = 0)
    
    return updated_center

def loss(centers, cluster, pixels):
    loss = 0
    e_dist = calc_distance(pixels, centers)
    for i in range(pixels.shape[0]):
        loss = loss + np.square(e_dist[i][cluster[i]])
    
    return loss

def main():
    img = cv2.imread('Q4image.png')
    image = img.copy()
    h ,w = image.shape[:2]
    image = image.reshape((h*w, 3))
    k = int(input('Enter the number of clusters'))
    iterations = int(input('Enter the number of iterations'))
    
    # initialize the centers
    centers = np.zeros((k,3))
    for i in range(k):
        rand_pixel = np.random.randint(h*w)
        centers[i] = image[rand_pixel]
    abs_tol=1e-16
    prev_loss = 0
    for i in range(iterations):
        assigned_cluster = assign_clusters(centers, image)
        centers = update_centers(image, assigned_cluster, centers, k)
        kmeans_loss = loss(centers, assigned_cluster, image)
        diff = np.abs(kmeans_loss - prev_loss)
        if (diff < abs_tol):
            break
        print('loss in iteration ', i)
        print(kmeans_loss)
        prev_loss = kmeans_loss

    segmented_img = image.copy()
    for i in range(0,k):
        cluster_indices = np.where(assigned_cluster == i)[0]
        segmented_img[cluster_indices] = centers[i]
        
    segmented_img = segmented_img.reshape(h,w,3)

    cv2.imwrite('k_means.png', segmented_img)
    cv2.imshow('img', segmented_img)
    cv2.waitKey(0)
if __name__ == '__main__':
    main()