import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    img = cv2.imread('../resources/target0.JPG')   # query image
    img1 = cv2.imread('../resources/image4.JPG')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(sigma = 3, nOctaveLayers = 20, contrastThreshold = 0.05)
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray1, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img, keypoints_1, img1, keypoints_2, matches, img1, flags=2)
    # plt.imshow(img3),plt.show()

    # extracting matches index
    matches_query_index = []
    matches_train_index = []
    for index in np.arange(len(matches)):
        matches_query_index.append(matches[index].queryIdx)
        matches_train_index.append(matches[index].trainIdx)

    # extracting X_Y of matches index
    x_y_values_query = keypoints_1[0].convert(keypoints_1)
    # adding keypoint index
    x_y_values_query = np.c_[x_y_values_query,np.arange(len(x_y_values_query))]

    matches_x_y_values_query = np.zeros((len(matches_query_index),3), dtype=np.float32)
    i = 0
    for index in matches_query_index:
        matches_x_y_values_query[i,:] = x_y_values_query[index,:]
        i = i+1
        
    x_y_values_train = keypoints_2[0].convert(keypoints_2)
    # adding keypoint index
    x_y_values_train = np.c_[x_y_values_train,np.arange(len(x_y_values_train))]
    matches_x_y_values_train = np.zeros((len(matches_train_index),3), dtype=np.float32)
    i = 0
    for index in matches_train_index:
        matches_x_y_values_train[i,:] = x_y_values_train[index,:]
        i = i+1

    # calculating mean and std
    x_mean_query = np.mean(matches_x_y_values_query[:,0])
    y_mean_query = np.mean(matches_x_y_values_query[:,1])
    x_std_query = np.std(matches_x_y_values_query[:,0])
    y_std_query = np.std(matches_x_y_values_query[:,1])

    x_mean_train = np.mean(matches_x_y_values_train[:,0])
    y_mean_train = np.mean(matches_x_y_values_train[:,1])
    x_std_train = np.std(matches_x_y_values_train[:,0])
    y_std_train = np.std(matches_x_y_values_train[:,1])

    # filtering
    filter_array = []
    for index in np.arange(len(matches_x_y_values_query)):
        if abs(x_mean_query - matches_x_y_values_query[index, 0]) < x_std_query:
            filter_array.append(True)
        else:
            filter_array.append(False)

    filtered_matches_x_y_values_query = matches_x_y_values_query[filter_array]

    filter_array = []
    for index in np.arange(len(filtered_matches_x_y_values_query)):
        if abs(y_mean_query - filtered_matches_x_y_values_query[index, 1]) < y_std_query:
            filter_array.append(True)
        else:
            filter_array.append(False)

    filtered_matches_x_y_values_query = filtered_matches_x_y_values_query[filter_array]

    filter_array = []
    for index in np.arange(len(matches_x_y_values_train)):
        if abs(x_mean_train - matches_x_y_values_train[index, 0]) < x_std_train:
            filter_array.append(True)
        else:
            filter_array.append(False)

    filtered_matches_x_y_values_train = matches_x_y_values_train[filter_array]

    filter_array = []
    for index in np.arange(len(filtered_matches_x_y_values_train)):
        if abs(y_mean_train - filtered_matches_x_y_values_train[index, 1]) < y_std_train:
            filter_array.append(True)
        else:
            filter_array.append(False)

    filtered_matches_x_y_values_train = filtered_matches_x_y_values_train[filter_array]

    filtered_keypoints_index_query = filtered_matches_x_y_values_query[:,2]
    filtered_keypoints_index_train = filtered_matches_x_y_values_train[:,2]

    # filtering keypoints
    filtered_keypoint_query = ()
    filtered_keypoint_train = ()

    for index in filtered_keypoints_index_query:
        filtered_keypoint_query = filtered_keypoint_query + (keypoints_1[int(index)],)
        
    for index in filtered_keypoints_index_train:
        filtered_keypoint_train = filtered_keypoint_train + (keypoints_2[int(index)],)

    len(filtered_keypoints_index_train)

    # filtering descriptors
    filtered_descriptors_query = np.zeros((len(filtered_keypoints_index_query),128), dtype=np.float32)
    i = 0
    for index in filtered_keypoints_index_query:
        filtered_descriptors_query[i,:] = descriptors_1[int(index),:]
        i = i+1
        
    filtered_descriptors_train = np.zeros((len(filtered_keypoints_index_train),128), dtype=np.float32)
    i = 0
    for index in filtered_keypoints_index_train:
        filtered_descriptors_train[i,:] = descriptors_2[int(index),:]
        i = i+1

    matches = bf.match(descriptors_1,filtered_descriptors_train)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img, keypoints_1, img1, filtered_keypoint_train, matches, img1, flags=2)

    # plt.imshow(img3),plt.show()
    start = (int(x_mean_train-x_std_train), int(y_mean_train-y_std_train))
    stop =  (int(x_mean_train+x_std_train), int(y_mean_train+y_std_train))
    marked_img = cv2.cv2.circle(img1, (int(x_mean_train), int(y_mean_train)), int(np.mean((x_std_train, y_std_train))), color = (255, 0, 0))
    plt.imshow(marked_img)
    plt.show()
    
if __name__ == "__main__":
    main()
