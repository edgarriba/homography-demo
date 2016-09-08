import numpy as np
import lutorpy as lua
require('nn')

import cv2

def extract_patches(img, kpts):
    patches = []  
    for kp in kpts:
        sub = cv2.getRectSubPix(img, (int(kp.size*1.3), int(kp.size*1.3)), kp.pt)
        res = cv2.resize(sub, (32, 32), interpolation = cv2.INTER_CUBIC)
        m   = np.ones((32,32), dtype=np.uint8) * cv2.mean(res)[0]
        # subtract mean
        patch = (res - m)
        patches.append(patch)
    return np.asarray(patches)

def compute_descriptors(img, kpts, net):
    print 'Computing descriptor from ', len(kpts), ' kpts'

    N = len(kpts)
    # TODO: check codes for a bigger batch size.
    # Now it only works with size of one.
    batch_sz = 1

    # extract the patches given the keypoints
    patches = extract_patches(img, kpts)
    assert N == len(patches)

    patches_t = torch.fromNumpyArray(patches)
    patches_t._view(N,1,32,32)

    patches_t   = patches_t._split(batch_sz)
    descriptors = []

    for i in range(N / batch_sz):
        print patches_t[i]._size()

        # infere Torch network
        prediction_t = net._forward(patches_t[i]._float())
        
        # Cast TorchTensor to NumpyArray and append to results
        prediction = prediction_t.asNumpyArray()

        # add the current prediction to the buffer
        descriptors.append(prediction)
        
        #print descriptors
        #raw_input("Press Enter to continue...")

    return np.float32(np.asarray(descriptors))


MIN_MATCH_COUNT = 10

TORCH_FILE = '/home/eriba/software/bmva/networks/torch/tfeat/margin_star/tfeat_test.t7'

net = torch.load(TORCH_FILE)
print net

# start opencv stuff

cap = cv2.VideoCapture('/home/eriba/Videos/Webcam/roibos_video.webm')
#cap = cv2.VideoCapture(0)

# Initiate ORB detector
det = cv2.ORB_create()
#det = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

img = cv2.imread('/home/eriba/Pictures/Webcam/roibos.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kp1, des1 = det.detectAndCompute(img, None)
des1 = compute_descriptors(img, kp1, net)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print gray.shape

    # find the keypoints and the descriptors
    kp2, des2 = det.detectAndCompute(gray, None)
    des2 = compute_descriptors(img, kp2, net)
    print 'Desc1: ', des1.shape, ' type: ', des1.dtype
    print 'Desc2: ', des2.shape, ' type: ', des2.dtype
   
    #matches = flann.knnMatch(des1, des2, k=2)
    good = bf.match(des1, des2)
    print 'Good matches ', len(good)

    # store all the good matches as per Lowe's ratio test.
#    good = []
#    for m,n in matches:
#        if m.distance < 0.7*n.distance:
#            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    
        h,w = img.shape
        #h,w,d = img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
    
        img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    
    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    img3 = cv2.drawMatches(img, kp1, frame, kp2, good, None, **draw_params)

    # Display the resulting frame
    cv2.imshow('frame', img3)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cv2.destroyAllWindows()
