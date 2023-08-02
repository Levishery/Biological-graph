import sys
sys.path.append('../..')

import numpy as np
from matplotlib import pyplot as plt
from biologicalgraphs.utilities import dataIO
from biologicalgraphs.graphs.biological import edge_generation
# from biologicalgraphs.cnns.biological import edges
from biologicalgraphs.transforms import seg2seg
from biologicalgraphs.skeletonization import generate_skeletons
from biologicalgraphs.algorithms import lifted_multicut
import pandas as pd
import pickle
import math
import h5py
import shutil
import os
from cloudvolume import CloudVolume
from tqdm import tqdm
import cc3d


import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Biologicla parameters.")
    parser.add_argument('--dust_thresh', type=int, default=1500)
    parser.add_argument('--dist', type=int, default=500)
    args = parser.parse_args()
    return args


def readh5(filename, dataset=None):
    fid = h5py.File(filename, 'r')
    if dataset is None:
        dataset = list(fid)[0]
    return np.array(fid[dataset])


def write_meta(grid_size, prefix):
    content = """\
# resolution in nm
16x16x40
# segmentation filename
/none
# grid size
%s""" % grid_size

    # Write the content to the file
    if not os.path.exists('meta/fafb_%s.meta'):
        with open('meta/%s.meta' % prefix, 'w') as file:
            file.write(content)


def get_block_paths(cord_start, cord_end):
    # block_dict = pickle.load(open(block_dict_path, 'rb'))
    # block_root_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    # # block_root_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent_test_3000_reformat'
    # block_list = []
    # for block_name in block_dict.keys():
    #     block_path = os.path.join(block_root_path, 'connector_' + block_name + '.csv')
    #     block_list.append(block_path)
    block_root_path = '/braindat/lab/liusl/flywire/block_data/v2/30_percent'
    block_list = []
    start_x, start_y, start_z = fafb_to_block(cord_start[0], cord_start[1], cord_start[2])
    end_x, end_y, end_z = fafb_to_block(cord_end[0], cord_end[1], cord_end[2])

    # Nested loop to find all integer points inside the bounding box
    for x in range(start_x, end_x + 1):
        for y in range(start_y, end_y + 1):
            for z in range(start_z, end_z + 1):
                block_name = str(x) + '_' + str(y) + '_' + str(z)
                block_list.append(os.path.join(block_root_path, 'connector_' + block_name + '.csv'))

    return block_list

def fafb_to_block(x, y, z, return_pixel=False):
    '''
    (x,y,z):fafb坐标
    (x_block,y_block,z_block):block块号
    (x_pixel,y_pixel,z_pixel):块内像素号,其中z为帧序号,为了使得z属于中间部分,强制z属于[29,54)
    文件名：z/y/z-xxx-y-xx-x-xx
    '''
    x_block_float = (x + 17631) / 1736 / 4
    y_block_float = (y + 19211) / 1736 / 4
    z_block_float = (z - 15) / 26
    x_block = math.floor(x_block_float)
    y_block = math.floor(y_block_float)
    z_block = math.floor(z_block_float)
    x_pixel = (x_block_float - x_block) * 1736
    y_pixel = (y_block_float - y_block) * 1736
    z_pixel = (z - 15) - z_block * 26
    while z_pixel < 28:
        z_block = z_block - 1
        z_pixel = z_pixel + 26
    if return_pixel:
        return x_block, y_block, z_block, np.round(x_pixel), np.round(y_pixel), np.round(z_pixel)
    else:
        return x_block, y_block, z_block

def is_inside_bounding_box(cord, bounding_box):
    # Unpack the bounding box coordinates
    min_point, max_point = bounding_box

    # Check if each component of cord is inside the bounding box
    inside_x = min_point[0] < cord[0] < max_point[0]
    inside_y = min_point[1] < cord[1] < max_point[1]
    inside_z = min_point[2] < cord[2] < max_point[2]

    # Return True only if all three components are inside the bounding box
    return inside_x and inside_y and inside_z

def stat_biological_recall(potential_path, block_paths, cord_start, cord_end):
    edges = pd.read_csv(potential_path, header=None)
    total_positives = 0
    hit = 0
    for block_path in block_paths:
        if os.path.exists(block_path):
            print('block %s'%block_path)
            samples = pd.read_csv(block_path, header=None)
            positives_indexes = np.where(samples[5] > 1)
            query = list(samples[0][list(positives_indexes[0])])
            pos = list(samples[1][list(positives_indexes[0])])
            cords = list(samples[2][list(positives_indexes[0])])
            for i in tqdm(range(len(query))):
                cord = np.array(cords[i][1:-1].split(), dtype=np.float32)
                if is_inside_bounding_box(cord, (cord_start, cord_end)):
                    total_positives = total_positives + 1
                    potentials = list(edges[1][np.where(edges[0] == query[i])[0]])
                    if pos[i] in potentials:
                        hit = hit + 1
                        continue
                    potentials = list(edges[1][np.where(edges[0] == pos[i])[0]])
                    if query[i] in potentials:
                        hit = hit + 1
                        continue
    print('bilogical edge recall: ', hit / total_positives)
    print('total positive samples: ', total_positives)


def divide_bounding_box(bbox):
    # Calculate midpoints along each dimension
    mid_point = (bbox[0] + bbox[1]) / 2

    # Define the 8 small bounding boxes
    boxes = [
        # Front left-bottom box
        (bbox[0], mid_point),

        # Front right-bottom box
        (np.array([mid_point[0], bbox[0][1], bbox[0][2]]), np.array([bbox[1][0], mid_point[1], mid_point[2]])),

        # Front left-top box
        (np.array([bbox[0][0], mid_point[1], bbox[0][2]]), np.array([mid_point[0], bbox[1][1], mid_point[2]])),

        # Front right-top box
        (np.array([mid_point[0], mid_point[1], bbox[0][2]]), np.array([bbox[1][0], bbox[1][1], mid_point[2]])),

        # Back left-bottom box
        (np.array([bbox[0][0], bbox[0][1], mid_point[2]]), np.array([mid_point[0], mid_point[1], bbox[1][2]])),

        # Back right-bottom box
        (np.array([mid_point[0], bbox[0][1], mid_point[2]]), np.array([bbox[1][0], mid_point[1], bbox[1][2]])),

        # Back left-top box
        (np.array([bbox[0][0], mid_point[1], mid_point[2]]), np.array([mid_point[0], bbox[1][1], bbox[1][2]])),

        # Back right-top box
        (mid_point, bbox[1]),
    ]

    return np.array(boxes)


def get_candidate_csv(label_name, unknown_filename_h5, csv_path=None, start_cord=None):
    mapping = readh5(label_name, 'original')
    edges = readh5(unknown_filename_h5)
    if csv_path is None:
        csv_path = label_name.replace('.h5', '_result.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

    for edge in tqdm(edges):
        seg_start = mapping[edge[3] - 1]
        seg_candidate = mapping[edge[4] - 1]
        # cord = np.asarray([edge[0], edge[1], edge[2]]) - np.asarray([8, 64, 64])
        # upper_bound = np.asarray([volume.shape[2], volume.shape[1], volume.shape[0]]) - np.asarray([16, 128, 128])
        # cord = [np.clip(cord[0], 0, upper_bound[0]), np.clip(cord[1], 0, upper_bound[1]), np.clip(cord[2], 0, upper_bound[2])]
        cord = np.asarray([edge[2], edge[1], edge[0]]) * np.array([4, 4, 1]) + start_cord
        row = pd.DataFrame(
            [{'node0_segid': int(seg_start), 'node1_segid': int(seg_candidate), 'cord': cord, 'target': -1,
              'prediction': -1}])
        row.to_csv(csv_path, mode='a', header=False, index=False)
    return csv_path


# the prefix name corresponds to the meta file in meta/{PREFIX}.meta
args = get_args()
vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0')
bbox = pickle.load(open('/braindat/lab/liusl/flywire/fafb-public-skeletons/top_10_box.pickle', 'rb'))
cord_start = bbox[0] / np.array([4, 4, 40])
cord_end = bbox[1] / np.array([4, 4, 40])
boxes = divide_bounding_box([cord_start, cord_end])
index = 0
for box in boxes:
    cord_start_box1 = box[0]
    cord_end_box1 = box[1]
    cord_mid_box1 = (box[0][0] + box[1][0]) / 2
    box_s = []
    box_s.append([box[0], np.array([cord_mid_box1, box[1][1], box[1][2]])])
    box_s.append([np.array([cord_mid_box1, box[1][1], box[1][2]]), box[1]])
    for small_box in box_s:
        cord_start_box2 = small_box[0]
        cord_end_box2 = small_box[1]
        volume_ffn1 = vol_ffn1[cord_start_box2[0] / 4:cord_end_box2[0] / 4 + 1,
                      cord_start_box2[1] / 4:cord_end_box2[1] / 4 + 1, cord_start_box2[2]:cord_end_box2[2] + 1].astype(np.uint64)
        print(volume_ffn1.shape)
        volume_ffn1 = np.asarray(volume_ffn1)[:, :, :, 0]

        prefix_woindex = 'fafb_dust%s_dis%s_cc3d_'%(str(args.dust_thresh), str(args.dist))
        prefix = prefix_woindex + str(index)
        print(prefix)
        grid_size = '%sx%sx%s'%(str(volume_ffn1.shape[0]), str(volume_ffn1.shape[1]), str(volume_ffn1.shape[2]))
        write_meta(grid_size, prefix)

        # read the input segmentation data
        segmentation = volume_ffn1
        segmentation = np.transpose(segmentation, [2, 1, 0])

        # subset is either training, validation, or testing
        subset = 'testing'
        # generate the skeleton by getting high->low resolution mappings
        # prevent OOM
        # hist = dataIO.get_seg_volume(segmentation)
        # dust_thresh = 1000
        # dust = list(np.where(hist < dust_thresh)[0])
        print('remove dust and relabel')
        segmentation = cc3d.dust(segmentation, threshold=args.dust_thresh, connectivity=26, in_place=False)
        segmentation, mapping = dataIO.relabel(segmentation)

        dataIO.WriteH5File(mapping['original'],
                           '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/' + prefix + 'mapping.h5',
                           'original', compression=True)
        segmentation = segmentation.astype(np.int64)

        # # and running topological thinnings
        print('Downsample Mapping')
        seg2seg.DownsampleMapping(prefix, segmentation)
        print('Topological Thinning')
        generate_skeletons.TopologicalThinning(prefix, segmentation)
        print('Find Endpoint Vectors')
        generate_skeletons.FindEndpointVectors(prefix)
        #
        # # run edge generation function
        # edge_generation.GenerateEdges_test(prefix, segmentation, subset)
        # unknown_filename_h5 = edge_generation.GenerateEdges_test(prefix, segmentation, subset, dust=dust)
        unknown_filename_h5 = edge_generation.GenerateEdges_test(prefix, segmentation, subset)
        index += index
        print('write candidates')
        potential_path = get_candidate_csv(
            '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/' + prefix + 'mapping.h5',
            unknown_filename_h5, csv_path='/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/'+ prefix_woindex + '.csv', start_cord=cord_start_box2)
        # potential_path = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/fafb_dust3000_small_test_.csv'
        block_paths = get_block_paths(cord_start_box2, cord_end_box2)
        stat_biological_recall(potential_path, block_paths, cord_start_box2, cord_end_box2)