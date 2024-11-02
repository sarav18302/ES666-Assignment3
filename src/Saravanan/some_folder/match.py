import cv2
import numpy as np
import random

class matchers():
    def __init__(self):
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm =0, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    def match(self,i1,i2):
        
        # Getting SIFT features
        imageSet1 = self.getSIFTFeatures(i1)
        imageSet2 = self.getSIFTFeatures(i2)
        
        #Feature matching
        matches = self.flann.knnMatch(
            imageSet2['des'],
            imageSet1['des'],
            k =2)
        
        #Applying Ratio test to find Good matches
        good_matches =[]
        for i,(m,n) in enumerate(matches):
            if(m.distance) < (0.5*n.distance):
                good_matches.append((m.trainIdx,m.queryIdx))
        print("Size(Good Matches):", len(good_matches))

        #Estimating the Homography Matrix
        if len(good_matches) >=4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']
            
            matchedPointsCurrent = np.float32(
                [pointsCurrent[i].pt for (_,i) in good_matches]
                )
            matchedPointsPrev = np.float32(
                [pointsPrevious[i].pt for (i,_) in good_matches]
                )
            
            temp_H,inliers_curr,inliers_prev = self.ransac(matchedPointsCurrent,matchedPointsPrev,4)
            
            #re-calculating Homography only using inlier points
            H = self.calculateHomography(inliers_curr,inliers_prev) 

            return H
        return None
    
    
    def calculateHomography(self,current,previous):
        
        Alist = []
        for i in range(len(current)):
            p1 = np.matrix([current[i][0],current[i][1],1])
            p2 = np.matrix([previous[i][0],previous[i][1], 1])
        
            a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                  p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                  p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            Alist.append(a1)
            Alist.append(a2)

        matrixA = np.matrix(Alist)
        u, s, v = np.linalg.svd(matrixA)
        H = np.reshape(v[8], (3, 3))
        H = (1/H.item(8)) * H
        return H
    
    def ransac(self,current,previous, thresh):
        
        maxInliers_curr, maxInliers_prev =[],[]
        finalH = None
        random.seed(2)
        for i in range(1000):
            
            ## Picking 4 random point correspondences
            currFour = np.empty((0, 2))
            preFour = np.empty((0,2))
            for j in range(4):
                random_pt = random.randrange(0, len(current))
                curr = current[random_pt]
                pre = previous[random_pt]
                currFour = np.vstack((currFour,curr))
                preFour = np.vstack((preFour,pre))
            

            #call the homography function on those points
            h = self.calculateHomography(currFour,preFour)
            
            #Calculate the inliers count of the calculated Homography
            inliers_curr = []
            inliers_prev =[]
            for i in range(len(current)):
                d = self.geometricDistance(current[i],previous[i], h)
                if d < 10:
                    inliers_curr.append([current[i][0],current[i][1]])
                    inliers_prev.append([previous[i][0],previous[i][1]])

            if len(inliers_curr) > len(maxInliers_curr):
                maxInliers_curr = inliers_curr
                maxInliers_prev = inliers_prev
                finalH = h
            # print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

            if len(maxInliers_curr) > (len(current)*thresh):
                break
            
        return finalH, maxInliers_curr,maxInliers_prev

    ## Computing Geometrical Error 
    def geometricDistance(self,current,previous, h):

        p1 = np.transpose(np.matrix([current[0], current[1], 1]))
        estimatep2 = np.dot(h, p1)
        estimatep2 = (1/estimatep2.item(2))*estimatep2

        p2 = np.transpose(np.matrix([previous[0], previous[1], 1]))
        error = p2 - estimatep2
        return np.linalg.norm(error)
    
    #Computing SIFT Keypoints and Descriptors
    def getSIFTFeatures(self,im):
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        kp,des = self.sift.detectAndCompute(gray,None)
        print(len(kp),len(des))
        return {'kp':kp,'des':des}