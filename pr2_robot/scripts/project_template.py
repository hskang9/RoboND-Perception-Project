#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Passthrough filter
def passthrough_filter(in_cloud, filter_axis, axis_min, axis_max):

    # Start by creating a filter object
    passthrough_filter = in_cloud.make_passthrough_filter()
    # Pass through based on given axis
    passthrough_filter.set_filter_field_name(filter_axis)

    passthrough_filter.set_filter_limits(axis_min, axis_max)
    out_cloud = passthrough_filter.filter()
    return out_cloud

def outlier_filter(in_cloud, k, threshold):
    # Start by creating a filter object
    outlier_filter = in_cloud.make_statistical_outlier_filter()
    
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(k)
    
    # Set threshold scale factor
    x = threshold

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    out_cloud = outlier_filter.filter()

    return out_cloud

def voxel_downsampling_filter(in_cloud, LEAF_SIZE):
    # Start by creating a filter object
    voxel_grid_filter = in_cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size   
    # Experiment and find the appropriate size!
    LEAF_SIZE = LEAF_SIZE

    # Set the voxel (or leaf) size  
    voxel_grid_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    out_cloud = voxel_grid_filter.filter()    
    return out_cloud

def RansacFilter(in_cloud, max_distance):
    # Create the segmentation object
    seg_objects = in_cloud.make_segmenter()
    
    # Set the model you wish to fit
    seg_objects.set_model_type(pcl.SACMODEL_PLANE)
    seg_objects.set_method_type(pcl.SAC_RANSAC)
    
    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    seg_objects.set_distance_threshold(max_distance)
    

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers_objects, coefficients_objects = seg_objects.segment()

    return inliers_objects, coefficients_objects


def EuclideanClustering(in_cloud, tolerance, minClusterSize, maxClusterSize):
    white_cloud = XYZRGB_to_XYZ(in_cloud) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(minClusterSize)
    ec.set_MaxClusterSize(maxClusterSize)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    

    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cluster_cloud)
    
    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
        
    return cluster_indices   


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pointCloud = ros_to_pcl(pcl_msg)
   

    # TODO: Statistical Outlier Filtering
    outlier_filtered = outlier_filter(pointCloud, 50, 1.0)
    

    # TODO: Voxel Grid Downsampling
    vox_filtered = voxel_downsampling_filter(outlier_filtered, 0.01)
    
    
    # TODO: PassThrough Filter

    # Pass through Z-axis
    z_filtered = passthrough_filter(vox_filtered, 'z', 0.6, 1.0)
    
    # Pass through X-axis
    z_x_filtered = passthrough_filter(z_filtered, 'x', 0.35, 0.8)
    
    # Pass through Y-axis
    z_x_y_filtered = passthrough_filter(z_x_filtered, 'y', -0.45, 0.45)    
    
    passthrough_objects = z_x_y_filtered
    passthrough_pub.publish(pcl_to_ros(passthrough_objects))
    
    
    # TODO: RANSAC Plane Segmentation

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers_objects, coefficients_objects = RansacFilter(passthrough_objects, 0.02)
    
    # TODO: Extract inliers and outliers

    extracted_objects = passthrough_objects.extract(inliers_objects, negative=True)
    extracted_table = passthrough_objects.extract(inliers_objects, negative=False)
    
    # TODO: Euclidean Clustering
    cluster_indices = EuclideanClustering(extracted_objects, 0.02, 50, 2000)

    

    
# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_objects.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=False)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(extracted_objects[pts_list[0]])[0:3]
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)





# function to load parameters and request PickPlace service
def pr2_mover(object_list, cloud_table):

    # TODO: Initialize variables
    pick_labels = []
    pick_centroids = []
    
    test_scene_num = Int32()
    test_scene_num.data = 1
        
    dict_list = []
    
    yaml_filename = "output_{}.yaml".format(test_scene_num.data)
    
    
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map
#    joint_publisher.publish()


    print("Iterating over objects in the pick list,...")
    # TODO: Loop through the pick list
    for pick_object_param in object_list_param:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        pick_object_name = pick_object_param['name']
        pick_object_group = pick_object_param['group']
        
        print("Looking for [[{}]] to be placed in [[{}]] box.".format(pick_object_name, pick_object_group))
        
#        if(picked_objects.find(pick_object_name)):
#            print(":: {} :: has already been picked, moving on,.. ".format(pick_object_name))
#            continue
        
        object_name = String()   
#        print(pick_object_name.__class__)     
        object_name.data = str(pick_object_name)
        
        # Create a collision cloud that contains the table and all the detected objects except the object to be picked.!!
        collision_cloud = cloud_table

        # Index of the object to be picked in the `detected_objects` list
        pick_object_cloud = None
        for i, detected_object in enumerate(object_list):
            if(detected_object.label == pick_object_name):
                pick_object_cloud = detected_object.cloud
            else:
                collision_cloud = AddClouds(collision_cloud, ros_to_pcl(detected_object.cloud))
            
        if(pick_object_cloud == None):
            print("ERROR:::: {} not found in the detected object list".format(pick_object_name))
            continue
                            
                            
        # Publish the collision cloud
        collision_pub.publish(pcl_to_ros(collision_cloud))
            
        points_arr = ros_to_pcl(pick_object_cloud).to_array()
#        print("Converted to array...")
        pick_object_centroid = np.mean(points_arr, axis=0)[:3] 
        print("Centroid found : {}".format(pick_object_centroid))

        pick_labels.append(pick_object_name)
        pick_centroids.append(pick_object_centroid)

        # Create pick_pose for the object
        pick_pose = Pose()
        pick_pose.position.x = float(pick_object_centroid[0])
        pick_pose.position.y = float(pick_object_centroid[1])
        pick_pose.position.z = float(pick_object_centroid[2])
#        pick_pose.orientation = 
                                                
        # TODO: Create 'place_pose' for the object
        place_pose = Pose()
        if(pick_object_group == 'green'):
            place_pose.position.x =  0
            place_pose.position.y = -0.71
            place_pose.position.z =  0.605
        else:
            place_pose.position.x =  0
            place_pose.position.y =  0.71
            place_pose.position.z =  0.605
                
#        print("Place pose created,... {}")

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        if(pick_object_group == 'green'):
            arm_name.data = 'right'
        else:
            arm_name.data = 'left'
                    
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        
        dict_list.append(yaml_dict)
        
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            print("Creating service proxy,...")
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            print("Requesting for service reponse,...")
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)
            picked_objects.append(pick_object_name)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml(yaml_filename, dict_list)




if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('capture_node')


    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    passthrough_pub = rospy.Publisher("/passthrough", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model_1.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
   
    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
