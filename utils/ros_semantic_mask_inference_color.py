import numpy as np
import rospy 
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
import ros_numpy
import cv2

from semantic_dynamic_reconfigure import SemanticDynamicReconfigure

class SegmenationMask:
    def __init__(self):
        
        self.image_height_yolo = 320
        self.image_width_yolo = 320
        
        # d455 camera
        self.rgb_image_subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgbImageCallback)
        
        self.segmantation_image_publisher = rospy.Publisher("/ultralytics/mask/image", Image, queue_size=1)
        # self.segmantation_image_publisher_oak = rospy.Publisher("/ultralytics/mask/image_oak", Image, queue_size=1)
        
        rospy.set_param("/rl_policy/semantic_label", 1)
        
        # Threshold of yellow in HSV space 
        self.lower = np.array([20, 100, 100]) 
        self.upper = np.array([30, 255, 255])
        self.enable_tuning = False
        self.lower_hue = 20
        self.upper_hue = 30
        self.lower_saturation = 100
        self.upper_saturation = 255
        self.lower_value = 100
        self.upper_value = 255
        
        self.hsv = SemanticDynamicReconfigure("HSV")
        self.hsv.add_variable("lower_hue", "A lower hue",    20, 0,   180)
        self.hsv.add_variable("upper_hue", "An upper hue",    30, 0,   180)

        self.hsv.add_variable("lower_saturation", "A lower saturation",    100, 0,   255)
        self.hsv.add_variable("upper_saturation", "An upper saturation",    255, 0,   255)

        self.hsv.add_variable("lower_value", "A lower value",    100, 0,   255)
        self.hsv.add_variable("upper_value", "An upper value",    255, 0,   255)

        self.hsv.add_variable("tuning", "A tuning activation parameter",  False)

        # Start the server
        self.hsv.start(self.dyn_rec_callback)
    
    def dyn_rec_callback(self, config, level):
        
        self.lower_hue = config.lower_hue
        self.lower_saturation = config.lower_saturation
        self.lower_value = config.lower_value
        
        self.upper_hue = config.upper_hue
        self.upper_saturation = config.upper_saturation
        self.upper_value = config.upper_value
        
        self.enable_tuning = config.tuning
        
        return config
    
    def rgbImageCallback(self, msg):
        yolo_class = rospy.get_param("/rl_policy/semantic_label")
        
        rgb_image = ros_numpy.numpify(msg)
        
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        # hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        # hue -- the color type (such as red, blue, or yellow), low hue means red, high hue means blue, 0-180
        # saturation -- the intensity of the color, low saturation means gray, high saturation means vivid, 0-255
        # value -- the brightness of the color, low value means dark, high value means light, 0-255

        if not self.enable_tuning:
            if yolo_class == 1: 
                # blue bubble
                self.lower = np.array([20, 20, 60]) 
                self.upper = np.array([50, 180, 255])
                # yellow
                # self.lower = np.array([95, 105, 90]) 
                # self.upper = np.array([105, 255, 255])
                # red
                # self.lower = np.array([120, 90, 150]) 
                # self.upper = np.array([140, 255, 255])
            elif yolo_class == 2:
                # blue
                self.lower = np.array([0, 95, 0]) 
                self.upper = np.array([20, 255, 243])
                # blue bubble
                # self.lower = np.array([20, 20, 60]) 
                # self.upper = np.array([40, 180, 255])
            elif yolo_class == 3:
            # green
                self.lower = np.array([87, 118, 0]) 
                self.upper = np.array([106, 205, 255])
            elif yolo_class == -1:
                self.lower = np.array([180, 255, 255]) 
                self.upper = np.array([180, 255, 255])
        else:
            self.lower[0] = self.lower_hue
            self.lower[1] = self.lower_saturation
            self.lower[2] = self.lower_value
        
            self.upper[0] = self.upper_hue
            self.upper[1] = self.upper_saturation
            self.upper[2] = self.upper_value
        # # Threshold of blue in HSV space 
        # lower = np.array([50, 40, 50])
        # upper = np.array([130, 255, 255])
        
        # # Threshold of yellow in HSV space 
        # self.lower = np.array([20, 100, 100]) 
        # self.upper = np.array([30, 255, 255])
        
        # # Threshold of dark green in HSV space
        # lower = np.array([40, 30, 0])
        # upper = np.array([75, 255, 255])        
        
        # # Threshold of brown in HSV space
        # lower = np.array([10, 100, 10])
        # upper = np.array([20, 150, 200])
        
        # # Threshold of red in HSV space
        # lower = np.array([0, 100, 100])
        # upper = np.array([10, 255, 255])
        
        # # Threshold of black in HSV space
        # lower = np.array([0, 0, 0])
        # upper = np.array([180, 255, 30])
        
        # # Threshold of white in HSV space
        # lower = np.array([0, 0, 200])
        # upper = np.array([180, 25, 255])

        # Get the color mask
        mask_original = cv2.inRange(hsv, self.lower, self.upper)
        size = (15, 15)
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, size)
        min_image = cv2.erode(mask_original, kernel)
        mask = cv2.resize(min_image, dsize=(self.image_height_yolo, self.image_width_yolo), interpolation=cv2.INTER_AREA)
        
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
            # try:
            #     clear_semantic_map = rospy.ServiceProxy('/semantic_node/clear_map', Empty)
            #     clear_semantic_map()
            # except rospy.ServiceException as e:
            #     print("Service call failed: %s"%e)

def main():
    # Initialize ROS node
    rospy.init_node('segmentation_mask_ultralytics')
    segmentation_mask = SegmenationMask()
    rospy.spin()

if __name__ == "__main__":
    main()
