#!/usr/bin/env python

# Python libs
import math, sys, json

# numpy
import numpy as np

# OpenCV
import cv2

# Ros libraries
import rospy

# Ros Messages
from sensor_msgs.msg import Image

# bite detector
import bitefinder

# TODO: Refactor the common functionality between this and biteserver
class PlateMaskGenerator(object):

    def __init__(self, options = {}):
        self.VERBOSE = options.get("verbose", False)
        self.finder = bitefinder.PlateFinder(options)
        self.ransac_iters = options.get("ransac_iters", 200)
        self.ransac_thresh = options.get("ransac_thresh", 0.01)  # 1cm thresh?
        self.plane_coeffs = [0.0, 0.0, 0.0]
        self.mask_filename = options.get("mask_filename", "mask.png")
        self.downscale_factor = options.get("downscale_factor", 0.5)
        self.save_depth_images = options.get("save_depth_images", False)
        self.depth_image_path = options.get("depth_image_path", "depth_images/")
        self.depth_image_format = options.get("depth_image_format", "png")
        self.mask_generated = False

    def decode_uncompressed_f32(self, data):
        if self.VERBOSE:
            print("Data has encoding: %s" % data.encoding)
        rows = data.height
        cols = data.step / 4  # assume 32 bit depth
        temp = np.fromstring(data.data, dtype=np.float32)
        temp = np.nan_to_num(temp)
        temp = temp.reshape(rows, cols)

        if self.downscale_factor < 1.0:
            cols = int(cols * self.downscale_factor)
            rows = int(rows * self.downscale_factor)
            temp = cv2.resize(temp, (cols, rows))

        if self.VERBOSE:
            print("rows: %d" % rows)
            print("cols: %d" % cols)
            print("max: %g" % np.max(temp))
            print("min: %g" % np.min(temp))

        if self.save_depth_images:
            fn = "{}frame_{}.{}".format(self.depth_image_path, 
                                       0, 
                                       self.depth_image_format)
            if self.depth_image_format == "png" or self.depth_image_format == "jpg":
                img = bitefinder.squash_depth(temp)
                color_img = cv2.merge([img, img, img])
                cv2.imwrite(fn, color_img)
            elif self.depth_image_format == "npy":
                np.save(fn, temp)
            else:
                print("Unknown depth image format: {}".format(self.depth_image_format))
                self.save_depth_images = False           

        return temp

    def process_depth(self, img):
        # first, fit a plane with ransac and get the residuals
        # best_coeffs, num_inliers, residuals = bitefinder.ransac_plane(img,
        #                                                               self.ransac_iters,
        #                                                               self.ransac_thresh,
        #                                                               self.plane_coeffs)
        best_coeffs, num_inliers, residuals = bitefinder.ransac_quad(img, 
                                                        self.ransac_iters, 
                                                        self.ransac_thresh)

        if self.VERBOSE:
            print("Number of inliers: {}".format(num_inliers))

        self.plane_coeffs = best_coeffs

        if(num_inliers == 0 or residuals is None):
            if(self.VERBOSE):
                print("No plane found.")
            return

        cv2.imwrite("plane_residuals.png", bitefinder.colorize_kernel(residuals * 0.1))
        mask = self.finder.build_plate_mask(residuals, -self.ransac_thresh)
        cv2.imwrite(self.mask_filename, cv2.merge((mask, mask, mask)))

        print("Mask generated.")
        rospy.signal_shutdown("Mask generated.")


    def callback_depth(self, data):
        img_base = self.decode_uncompressed_f32(data)
        self.process_depth(img_base)

    def start_listening(self, depth_topic):
        rospy.init_node('platemaskgenerator')

        self.depth_sub = rospy.Subscriber(depth_topic, Image,
                                          self.callback_depth, queue_size=1)


def load_options(optlist):
    opts = {}
    for fn in optlist:
        with open(fn, "rt") as src:
            temp_opts = json.load(src)
            opts.update(temp_opts)
    if opts.get("verbose", False):
        print("Options: {}".format(opts))
    return opts

if __name__ == '__main__':

    if len(sys.argv) > 1:
        optfns = sys.argv[1:]
        opts = load_options(optfns)
    else:
        opts = {}

    frame_listener = PlateMaskGenerator(opts)

    depth_topic = opts.get("depth_topic", "/camera/depth/image")

    test_depth_image = opts.get("test_image", "")
    if test_depth_image != "":
        print("Running a test image...")
        image_base = np.load(test_depth_image)
        frame_listener.process_depth(image_base)
        print("Finished running test.")
    else:
        frame_listener.start_listening(depth_topic)
        # keep ros going
        rospy.spin()