# import cv2
import numpy as np
import torch
from ultralytics import YOLO
import random
import rospy 
import math
from sensor_msgs.msg import Image
import ros_numpy
import torch.nn.functional as F

import cv2

# Fast SAM
# from ultralytics import FastSAM
# from ultralytics.models.fastsam import FastSAMPrompt


class SegmenationMask:
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_height_yolo = 320
        self.image_width_yolo = 320
        
        # self.segmentation_model = YOLO("yolo11s-seg.pt")
        # self.segmentation_model.export(format="engine", imgsz=(self.image_height_yolo, self.image_width_yolo), half=True)
        self.tensorrt_model = YOLO("yolo11s-seg.engine")
        
        # d455 camera
        self.rgb_image_subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgbImageCallback)
        
        self.segmantation_image_publisher = rospy.Publisher("/ultralytics/mask/image", Image, queue_size=1)
        
        rospy.set_param("/rl_policy/semantic_label", 28)
    
    def rgbImageCallback(self, msg):
        # Semantic detector and mask
        ## Classes
        ## 1: bicycle
        ## 24: backpack
        ## 28: suitcase
        ## 56: chair
        ## 57: couch
        ## 60: dining table
        ## 62: tv
        ## 74: clock
        yolo_class = rospy.get_param("/rl_policy/semantic_label")
        
        rgb_image = ros_numpy.numpify(msg)
        self.rgb_image = cv2.resize(rgb_image, dsize=(self.image_height_yolo, self.image_width_yolo), interpolation=cv2.INTER_CUBIC)
        seg_results = self.tensorrt_model(self.rgb_image, classes=yolo_class, verbose=False, imgsz=(self.image_height_yolo, self.image_width_yolo), conf=0.5, device=self.device)
        mask = np.zeros((self.image_height_yolo, self.image_width_yolo)).astype('float32')
        if seg_results[0].masks is not None:
            for result in seg_results:
                mask += (self.min_filter(result.masks.data[0], kernel_size=21).cpu().numpy() * 255).astype('float32')
                # mask += (result.masks.data[0].cpu().numpy() * 255).astype('float32')
        # oak_mask = np.where(mask == 255.0, 255.0, 0.0).astype('uint8')
        yolo_class_new = rospy.get_param("/rl_policy/semantic_label")
        if yolo_class_new == yolo_class:
          mask = np.where(mask == 255.0, 2.0, 1.0).astype('float32')
        else:
          mask = np.where(mask == 255.0, 1.0, 1.0).astype('float32')
        yolo_class_new = rospy.get_param("/rl_policy/semantic_label")
        if yolo_class_new == yolo_class:
            msg_pub = ros_numpy.msgify(Image, mask, encoding="32FC1")
            msg_pub.header = msg.header
            self.segmantation_image_publisher.publish(msg_pub)
        else:
            mask = np.where(mask == 255.0, 1.0, 1.0).astype('float32')
            msg_pub = ros_numpy.msgify(Image, mask, encoding="32FC1")
            msg_pub.header = msg.header
            self.segmantation_image_publisher.publish(ros_numpy.msgify(Image, mask, encoding="32FC1"))

        # self.segmantation_image_publisher_oak.publish(msg)
    def min_filter(self, image, kernel_size=3):
        padding = kernel_size // 2

        # Pad the image
        padded_image = torch.nn.functional.pad(image.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect').squeeze(0)

        # Unfold the image into sliding windows
        unfolded = torch.nn.functional.unfold(padded_image.unsqueeze(0), kernel_size=(kernel_size, kernel_size))
        unfolded = unfolded.squeeze(0).transpose(0, 1)

        # Mask for valid values (not -10.0 or -1.0)
        valid_pixels = (unfolded != -1.0)

        # Replace invalid pixels (-10.0) with NaN for median calculation
        valid_unfolded = torch.where(valid_pixels, unfolded, torch.tensor(np.nan, device=self.device))

        # Compute the median ignoring NaN values
        min_values = torch.min(valid_unfolded, dim=1).values

        # Reshape the median values back to the image shape
        min_image = min_values.view(image.shape)
        # median_image = torch.nan_to_num(median_image, nan=-1.0)
        
        # Replace -1.0 pixels with their corresponding median values, but leave -10.0 unchanged
        # filtered_image = torch.where(image == 1.0, min_image, image)

        return min_image

def main():
    # Initialize ROS node
    rospy.init_node('segmentation_mask_ultralytics')
    segmentation_mask = SegmenationMask()
    rospy.spin()

if __name__ == "__main__":
    main()
