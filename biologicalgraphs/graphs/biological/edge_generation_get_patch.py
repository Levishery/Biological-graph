# coding=utf-8
import math
import os
import time
import random
import struct
import pandas as pd

import numpy as np
from numba import jit

from biologicalgraphs.utilities import dataIO
from biologicalgraphs.utilities.constants import *
from biologicalgraphs.graphs.biological.util import CreateDirectoryStructure, ExtractExample, GenerateAllExamplesArray, ScaleFeature

# 1 dir changed



@jit(nopython=True)
def ExtractAdjacencyMatrix(segmentation):
    zres, yres, xres = segmentation.shape

    # create a set of neighbors as a tuple with the lower label first
    # if (z, y, x) is 1, the neighbor +1 is a different label
    xdiff = segmentation[:,:,1:] != segmentation[:,:,:-1]
    ydiff = segmentation[:,1:,:] != segmentation[:,:-1,:]
    zdiff = segmentation[1:,:,:] != segmentation[:-1,:,:]

    adjacency_graph = set()

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if iz < zres - 1 and zdiff[iz,iy,ix]:
                    adjacency_graph.add((segmentation[iz,iy,ix], segmentation[iz+1,iy,ix]))
                if iy < yres - 1 and ydiff[iz,iy,ix]:
                    adjacency_graph.add((segmentation[iz,iy,ix], segmentation[iz,iy+1,ix]))
                if ix < xres - 1 and xdiff[iz,iy,ix]:
                    adjacency_graph.add((segmentation[iz,iy,ix], segmentation[iz,iy,ix+1]))

    # make sure that label_one is less than label_two to avoid double edges
    corrected_adjacency_graph = set()
    for (label_one, label_two) in adjacency_graph:
        if not label_one or not label_two: continue
        if label_two < label_one: corrected_adjacency_graph.add((label_two, label_one))
        else: corrected_adjacency_graph.add((label_one, label_two))

    return corrected_adjacency_graph



def BaselineGraph(prefix, segmentation, seg2gold_mapping):
    # get the adjacency matrix
    adjacency_graph = ExtractAdjacencyMatrix(segmentation)

    positive_candidates = []
    negative_candidates = []
    unknown_candidates = []

    for (label_one, label_two) in adjacency_graph:
        gold_one = seg2gold_mapping[label_one]
        gold_two = seg2gold_mapping[label_two]

        if gold_one < 1 or gold_two < 1: unknown_candidates.append((label_one, label_two))
        elif gold_one == gold_two: positive_candidates.append((label_one, label_two))
        else: negative_candidates.append((label_one, label_two))

    print ('Baseline Adjacency Graph Results')
    print ('  Number positive edges {}'.format(len(positive_candidates)))
    print ('  Number negative edges {}'.format(len(negative_candidates)))
    print ('  Number unknowns edges {}'.format(len(unknown_candidates)))

    baseline_filename = 'edge-baselines/{}-edge-baselines.txt'.format(prefix)
    with open(baseline_filename, 'w') as fd:
        fd.write('Baseline Adjacency Graph Results\n')
        fd.write('  Number positive edges {}\n'.format(len(positive_candidates)))
        fd.write('  Number negative edges {}\n'.format(len(negative_candidates)))
        fd.write('  Number unknowns edges {}\n'.format(len(unknown_candidates)))




@jit(nopython=True)
def TraverseIndividualEndpoint(segmentation, center, vector, resolution, max_label, maximum_distance):
    # the maximum degrees is a function of how the endpoint vectors are generated
    # the vectors have at best this resolution accuracy
    maximum_radians = 0.3216   # ~约等于 18.5°
    # save computation time by calculating cos(theta) here
    cos_theta = math.cos(maximum_radians)

    # decompress important variables
    zpoint, ypoint, xpoint = center
    zradius, yradius, xradius = (int(maximum_distance / resolution[IB_Z]), int(maximum_distance / resolution[IB_Y]), int(maximum_distance / resolution[IB_X]))

    zres, yres, xres = segmentation.shape

    # cur
    label = segmentation[zpoint,ypoint,xpoint]

    # # create a set of labels to ignore
    labels_to_ignore = set()
    # start by ignoring all labels with the same value
    labels_to_ignore.add(segmentation[zpoint,ypoint,xpoint])

    # keep track of what is adjacent in this cube and which potential neighbors are already on the stack
    adjacency_matrix = set()
    potential_neighbors = set()

    zmeans = np.zeros(max_label, dtype=np.float32)
    ymeans = np.zeros(max_label, dtype=np.float32)
    xmeans = np.zeros(max_label, dtype=np.float32)
    counts = np.zeros(max_label, dtype=np.float32)

    # iterate through the window
    for iz in range(zpoint - zradius, zpoint + zradius + 1):
        if iz < 0 or iz > zres - 1: continue
        for iy in range(ypoint - yradius, ypoint + yradius + 1):
            if iy < 0 or iy > yres - 1: continue
            for ix in range(xpoint - xradius, xpoint + xradius + 1):
                if ix < 0 or ix > xres - 1: continue
                # get the  label for this location
                voxel_label = segmentation[iz,iy,ix]

                # skip over extracellular/unlabeled material
                if not voxel_label: continue

                # update the adjacency matrix
                if iz < zres - 1 and voxel_label != segmentation[iz+1,iy,ix]:
                    adjacency_matrix.add((voxel_label, segmentation[iz+1,iy,ix]))

                    # update mean affinities
                    if voxel_label == label or segmentation[iz+1,iy,ix] == label:
                        index = segmentation[iz+1,iy,ix] if voxel_label == label else voxel_label

                        zmeans[index] += (iz + 0.5)
                        ymeans[index] += iy
                        xmeans[index] += ix
                        counts[index] += 1

                if iy < yres - 1 and voxel_label != segmentation[iz,iy+1,ix]:
                    adjacency_matrix.add((voxel_label, segmentation[iz,iy+1,ix]))

                    # update mean affinities
                    if voxel_label == label or segmentation[iz,iy+1,ix] == label:
                        index = segmentation[iz,iy+1,ix] if voxel_label == label else voxel_label

                        zmeans[index] += iz
                        ymeans[index] += (iy + 0.5)
                        xmeans[index] += ix
                        counts[index] += 1

                if ix < xres - 1 and voxel_label != segmentation[iz,iy,ix+1]:
                    adjacency_matrix.add((voxel_label, segmentation[iz,iy,ix+1]))

                    # update mean affinities
                    if voxel_label == label or segmentation[iz,iy,ix+1] == label:
                        index = segmentation[iz,iy,ix+1] if voxel_label == label else voxel_label

                        zmeans[index] += iz
                        ymeans[index] += iy
                        xmeans[index] += (ix + 0.5)
                        counts[index] += 1


                # skip points that belong to the same label
                # needs to be after adjacency lookup
                if voxel_label in labels_to_ignore: continue

                # find the distance between the center location and this one and skip if it is too far
                zdiff = resolution[IB_Z] * (iz - zpoint)
                ydiff = resolution[IB_Y] * (iy - ypoint)
                xdiff = resolution[IB_X] * (ix - xpoint)

                distance = math.sqrt(zdiff * zdiff + ydiff * ydiff + xdiff * xdiff)
                if distance > maximum_distance: continue

                # get a normalized vector between this point and the center
                vector_to_point = (zdiff / distance, ydiff / distance, xdiff / distance)

                # get the distance between the two vectors
                dot_product = vector[IB_Z] * vector_to_point[IB_Z] + vector[IB_Y] * vector_to_point[IB_Y] + vector[IB_X] * vector_to_point[IB_X]

                # get the angle from the dot product
                if (dot_product < cos_theta): continue

                # add this angle to the list to inspect further and ignore it every other time
                labels_to_ignore.add(voxel_label)
                potential_neighbors.add(voxel_label)

    # only include potential neighbor labels that are locally adjacent
    neighbors = []
    means = []

    for neighbor_label in potential_neighbors:
        # do not include background predictions
        if not neighbor_label: continue
        # make sure that the neighbor is locally adjacent and add to the set of edges
        if not (neighbor_label, label) in adjacency_matrix and not (label, neighbor_label) in adjacency_matrix: continue
        neighbors.append(neighbor_label)

        # return the mean as integer values and continue
        means.append((int(zmeans[neighbor_label] / counts[neighbor_label]), int(ymeans[neighbor_label] / counts[neighbor_label]), int(xmeans[neighbor_label] / counts[neighbor_label])))

    return neighbors, means



def EndpointTraversal(prefix, segmentation, maximum_distance):
    # get the resolution for this data
    resolution = dataIO.Resolution(prefix)

    # get the maximum label
    max_label = np.amax(segmentation) + 1

    # read in all of the skeletons
    skeletons = dataIO.ReadSkeletons(prefix)

    # create a set of labels to consider
    edges = []

    # go through every skeletons endpoints
    for skeleton in skeletons:
        label = skeleton.label  # 当前segment的label

        for ie, endpoint in enumerate(skeleton.endpoints):
            # get the (x, y, z) location
            # 遍历当前segment中的端点以及端点向量
            center = (endpoint.iz, endpoint.iy, endpoint.ix)
            vector = endpoint.vector
            # do not consider null vectors (the sums are all 0 or 1)
            if vector[IB_Z] * vector[IB_Z] + vector[IB_Y] * vector[IB_Y] + vector[IB_X] * vector[IB_X] < 0.5: continue

            # 获取当前端点的neigbors（相邻segment的label) 以及端点到该segment的边界中心点
            # 遍历每个端点maximum_distance范围内邻居voxel，每个端点都有一个labels_to_ignore 防止重复添加
            neighbors, means = TraverseIndividualEndpoint(segmentation, center, vector, resolution, max_label, maximum_distance)

            for iv, neighbor_label in enumerate(neighbors):
                (zpoint, ypoint, xpoint) = means[iv]
                # append this to this list of edges
                edges.append((zpoint, ypoint, xpoint, label, neighbor_label, ie))  # ie是指当前端点在该segment上的序号


    return edges



def GenerateEdges(prefix, segmentation, subset, seg2gold_mapping, gold, image):
    # parameters from CVPR submission
    network_radius = 0     # 寻找candidate的cube 半径
    maximum_distance = 500   # 端点寻找的距离
    width = (8, 64, 64)   # #  cnn 训练cube

    # create the directory structure to save the features in
    # forward is needed for training and validation data that is cropped
    CreateDirectoryStructure(width, network_radius, ['training', 'validation', 'testing', 'forward'], 'edges')

    # get the size of the data
    zres, yres, xres = segmentation.shape

    # make sure the subset is one of three categories
    assert (subset == 'training' or subset == 'validation' or subset == 'testing')

    # crop the subset if it overlaps with testing data
    ((cropped_zmin, cropped_zmax), (cropped_ymin, cropped_ymax), (cropped_xmin, cropped_xmax)) = dataIO.CroppingBox(prefix)

    # call the function to actually generate the edges
    # find edges(z,y,x, label_one, label_two)
    edges = EndpointTraversal(prefix, segmentation, maximum_distance)

    # create list for all relevant examples
    positive_examples = []
    negative_examples = []
    unknown_examples = []
    forward_positive_examples = []
    forward_negative_examples = []
    forward_unknown_examples = []

    for edge in edges:
        zpoint, ypoint, xpoint = (edge[IB_Z], edge[IB_Y], edge[IB_X])
        label_one, label_two = edge[3], edge[4]

        # if the center of the point falls outside the cropped box do not include it in training or validation
        forward = False
        # however, you allow it for forward inference
        if (zpoint < cropped_zmin or cropped_zmax <= zpoint): forward = True
        if (ypoint < cropped_ymin or cropped_ymax <= ypoint): forward = True
        if (xpoint < cropped_xmin or cropped_xmax <= xpoint): forward = True

        # see if these two segments belong to the same neuron
        if not seg2gold_mapping is None:
            gold_one = seg2gold_mapping[label_one]
            gold_two = seg2gold_mapping[label_two]
        else:
            gold_one = -1
            gold_two = -1

        # create lists of locations where these point occur
        if forward:
            if gold_one < 1 or gold_two < 1:
                forward_unknown_examples.append(edge)
            elif gold_one == gold_two:
                forward_positive_examples.append(edge)
            else:
                forward_negative_examples.append(edge)
        else:
            if gold_one < 1 or gold_two < 1:
                unknown_examples.append(edge)
            elif gold_one == gold_two:
                positive_examples.append(edge)
            else:
                negative_examples.append(edge)

    print ('No. Positive Edges: {}'.format(len(positive_examples)))
    print ('No. Negative Edges: {}'.format(len(negative_examples)))
    print ('No. Unknown Edges: {}'.format(len(unknown_examples)))

    parent_directory = '/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/edges-{}nm-{}x{}x{}'.format(network_radius, width[IB_Z], width[IB_Y], width[IB_X])
    csv_filename = '{}/{}/{}.csv'.format(parent_directory, subset, prefix)
    targets = []
    examples = []
    examples_positive = []
    examples_negative = []

    if len(positive_examples):
        # save the examples
        positive_filename = '{}/{}/positives/{}.examples'.format(parent_directory, subset, prefix)
        with open(positive_filename, 'wb') as fd:
            fd.write(struct.pack('q', len(positive_examples)))
            for ie, example in enumerate(positive_examples):
                fd.write(struct.pack('qqqqqq', example[0], example[1], example[2], example[3], example[4], example[5]))

        # create new examples array to remove last element
        for example in positive_examples:
            examples.append(list(example[0:5]))
            examples_positive.append(list(example[0:5]))
            targets.append(1)

        positive_examples_array, positive_examples_gt_array, positive_examples_img_array = GenerateAllExamplesArray(prefix, segmentation, examples_positive, width, gold, image)
        dataIO.WriteH5File(positive_examples_array, '{}/{}/positives/{}-examples.h5'.format(parent_directory, subset, prefix), 'main', compression=True)
        dataIO.WriteH5File(positive_examples_gt_array,
                           '{}/{}/positives/{}-examples-gt.h5'.format(parent_directory, subset, prefix), 'main',
                           compression=True)
        dataIO.WriteH5File(positive_examples_img_array,
                           '{}/{}/positives/{}-examples-img.h5'.format(parent_directory, subset, prefix), 'main',
                           compression=True)
        del positive_examples_array
        del positive_examples_gt_array
        del positive_examples_img_array

    if len(negative_examples):
        # save the examples
        negative_filename = '{}/{}/negatives/{}.examples'.format(parent_directory, subset, prefix)
        with open(negative_filename, 'wb') as fd:
            fd.write(struct.pack('q', len(negative_examples)))
            for ie, example in enumerate(negative_examples):
                fd.write(struct.pack('qqqqqq', example[0], example[1], example[2], example[3], example[4], example[5]))

        for example in negative_examples:
            examples.append(list(example[0:5]))
            examples_negative.append(list(example[0:5]))
            targets.append(0)

        negative_examples_array, negative_examples_gt_array, negative_examples_img_array = GenerateAllExamplesArray(prefix, segmentation, examples_negative, width, gold, image)
        dataIO.WriteH5File(negative_examples_array, '{}/{}/negatives/{}-examples.h5'.format(parent_directory, subset, prefix), 'main', compression=True)
        dataIO.WriteH5File(negative_examples_gt_array,
                           '{}/{}/negatives/{}-examples-gt.h5'.format(parent_directory, subset, prefix), 'main',
                           compression=True)
        dataIO.WriteH5File(negative_examples_img_array,
                           '{}/{}/negatives/{}-examples-img.h5'.format(parent_directory, subset, prefix), 'main',
                           compression=True)
        del negative_examples_array
        del negative_examples_gt_array
        del negative_examples_img_array

    df = pd.DataFrame(
        columns=['label_one', 'label_two', 'probabilities', 'predictions', 'target', 'xmean', 'ymean', 'zmean', 'ie',
                 'iex', 'iey', 'iez', 'ie_vector'])
    vertex_ones = list(np.array(examples)[:,3])
    vertex_twos = list(np.array(examples)[:,4])
    df['label_one'] = vertex_ones
    df['label_two'] = vertex_twos
    df['probabilities'] = len(vertex_ones)*[0]
    df['predictions'] = len(vertex_ones)*[0]
    df['target'] = targets
    df['xmean'] = list(np.array(examples)[:, IB_X])
    df['ymean'] = list(np.array(examples)[:, IB_Y])
    df['zmean'] = list(np.array(examples)[:, IB_Z])
    df.to_csv(os.path.join(parent_directory, csv_filename))
