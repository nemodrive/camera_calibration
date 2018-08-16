#!/usr/bin/python
import numpy as np
import cv2
import glob
import sys, getopt
import json

file = open('log_far.txt', 'w')


def save_photos(path, camera_id):
    cap = cv2.VideoCapture(int(camera_id))
    contor = 0
    while True:
        print('pls work')
        _, frame = cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            string = path + '/image_' + str(contor) + '.png'
            print(string)
            cv2.imwrite(string, frame)
            contor = contor + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def calibrate(path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (6,5,0)
    objp = np.zeros((9 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    imgpoints = []  # 2d points in image plane
    objpoints = []  # 3d point in real world space

    images = glob.glob(path + '/*.png')

    print(images)

    data = {}
    contor = 0
    name = 'logitech_2'
    data['name'] = name
    with open(path + '/log.json', 'w') as log:
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)
            print(ret)
            # If found, add object points, image points (after refining them)
            if ret == True:
                contor = contor + 1
                data['photo_no'] = fname
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                frame = cv2.drawChessboardCorners(img, (7, 9), corners2, ret)
                cv2.imshow('frame', frame)
                # cv2.waitKey(500)
                # data['frame'] = frame.tolist()
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                data['ret'] = ret
                data['mtx'] = mtx.tolist()
                data['dist'] = dist.tolist()
                data['rvecs'] = [rvecs[0].tolist()]
                data['tvecs'] = [tvecs[0].tolist()]
                print(data)
                # exit(0)
                json.dump(data, log, indent=4)
    log.close()
    cv2.destroyAllWindows()
    mean_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print ("total error: ", mean_error / len(objpoints))


def undist(path_to_photo):
    img = cv2.imread(path_to_photo)
    h, w = img.shape[:2]
    newcameramtx, rio = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h),l, (w,h))
    print(newcameramtx, rio)


def view_only(cameraID):
    cap = cv2.VideoCapture(int(cameraID))

    while True:
        _, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hvs", ["view-only=", "help=", "calibration="])
        print(opts, args)
    except getopt.GetoptError:
        print ('intrinsic_calibration.py <path> <camera_id>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h',"--help"):
            print ('intrinsic_calibration.py <path> <camera_id>')
            sys.exit()
        elif opt in ('-v','--view-only'):
            view_only(args[0])
            sys.exit()
        elif opt in ('-s', '--save'):
            save_photos(args[0], args[1])
            exit(0)
    if len(args) != 1:
        sys.exit()
    print args[0]
    calibrate(args[0])
    sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])
