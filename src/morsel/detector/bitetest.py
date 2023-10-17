from bitefinder import *
import numpy as np
import cv2
import json
import sys

def basic_test(test_image, options):
    testo = BiteFinder(options)
    cv2.imwrite("test.png", colorize_kernel(testo._kernel))

    inlier_thresh = 0.02

    #best_coeffs, num_inliers, residuals = ransac_plane(test_image, 200, inlier_thresh)
    best_coeffs, num_inliers, residuals = ransac_quad(test_image, 200, inlier_thresh)
    print("Best coeffs: %s" % str(best_coeffs))
    print("Num inliers: %d" % num_inliers)

    cv2.imwrite("plane_residuals.png", colorize_kernel(residuals * 0.1))

    jdata = {}
    jdata["plane"] = list(best_coeffs)

    binimage = np.zeros(residuals.shape, dtype=np.uint8)
    binimage[residuals < -inlier_thresh] = 255
    blurimage = cv2.GaussianBlur(binimage, (7,7), 2.0)
    binimage[blurimage > 64] = 255
    binimage[blurimage <= 64] = 0
    cv2.imwrite("bin_residuals.png", binimage)
    circs = cv2.HoughCircles(binimage, cv2.cv.CV_HOUGH_GRADIENT, 1, 40, param2=20)
    print("Circles: " + str(circs))
    circimage = cv2.merge((binimage, binimage, binimage))
    for (x,y,rad) in circs[0]:
        cv2.circle(circimage, (x,y), rad, (255, 0, 255), 3)
    cv2.imwrite("circles.png", circimage)

    bites = testo.find_bites(residuals, 10, -inlier_thresh)
    print(bites)

    jdata["bites"] = bites

    with open("data.json", "wt") as dest:
        dest.write(json.dumps(jdata))

    cv2.imwrite("test_bites.png", testo._debug_image)
    cv2.imwrite("test_qual.png", colorize_kernel(testo._last_bite_quality * (1.0 / testo._kernel_size ** 2.0)))

def plate_test(test_image, options):
    pfinder = PlateFinder(options)
    inlier_thresh = 0.02

    best_coeffs, num_inliers, residuals = ransac_quad(test_image, 200, inlier_thresh)
    print("Best coeffs: %s" % str(best_coeffs))
    print("Num inliers: %d" % num_inliers)

    cv2.imwrite("plane_residuals.png", colorize_kernel(residuals * 0.1))
    mask = pfinder.build_plate_mask(residuals, -inlier_thresh)
    cv2.imwrite("mask.png", cv2.merge((mask, mask, mask)))

def load_options(optlist):
    opts = {"debug": True}
    for fn in optlist:
        with open(fn, "rt") as src:
            temp_opts = json.load(src)
            opts.update(temp_opts)
    print("Options: {}".format(opts))
    return opts

def run_tests(filename, options_list):
    test_image = np.load(filename)
    options = load_options(options_list)
    #basic_test(test_image, options)
    plate_test(test_image, options)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Must specify .npy test depth image!")
    else:
        run_tests(sys.argv[1], sys.argv[2:])
