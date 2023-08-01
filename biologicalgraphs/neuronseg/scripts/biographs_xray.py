import numpy as np
from matplotlib import pyplot as plt
from biologicalgraphs.utilities import dataIO
from biologicalgraphs.graphs.biological import edge_generation
# from biologicalgraphs.cnns.biological import edges
from biologicalgraphs.transforms import seg2seg
from biologicalgraphs.skeletonization import generate_skeletons
from biologicalgraphs.algorithms import lifted_multicut
from tqdm import tqdm
import shutil
import pandas as pd
import h5py
import shutil
import os


def readh5(filename, dataset=None):
    fid = h5py.File(filename, 'r')
    if dataset is None:
        dataset = list(fid)[0]
    return np.array(fid[dataset])

def stat_biological_recall(potential_path, positive_path):
    edges = pd.read_csv(potential_path, header=None)
    total_positives = 0
    hit = 0
    resolution_scale = 700/300
    hit_csv_path = potential_path.replace('mapping_result.csv', '_hit.csv')
    sample_dir = potential_path.replace('mapping_result.csv', 'samples')
    if os.path.exists(hit_csv_path):
        os.remove(hit_csv_path)
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.mkdir(sample_dir)
    samples = pd.read_csv(positive_path)
    query = samples['id1']
    pos = samples['id2']
    all_positive_segments = list(samples['id1']) + list(samples['id2'])
    for i in range(len(query)):
        total_positives = total_positives + 1
        potentials = list(edges[1][np.where(edges[0] == query[i])[0]])
        if pos[i] in potentials:
            hit = hit + 1
            row = pd.DataFrame([{'node0_segid': int(query[i]), 'node1_segid': int(pos[i]), 'hit': 1}])
            row.to_csv(hit_csv_path, mode='a', header=False, index=False)
            # get positive and negative samples
            sample_path = os.path.join(sample_dir, str(query[i]) + '_' + str(pos[i]) + '.csv')
            potential_edges = list(np.where(edges[0] == query[i])[0]) + list(np.where(edges[0] == pos[i])[0])
            idx = np.where(edges[0] == query[i])[0][np.where(pos[i]==potentials)[0][0]]
            cord = np.fromstring(edges[2][idx][2:-1], dtype=int, sep=' ')
            cord = [int(cord[0]*resolution_scale+250), int(cord[1]*resolution_scale+250), int(cord[2]*resolution_scale+250)]
            row = pd.DataFrame([{'node0_segid': int(query[i]), 'node1_segid': int(pos[i]), 'cord':cord, 'target': 1, 'prediction': -1}])
            row.to_csv(sample_path, mode='a', header=False, index=False)
            for potential_edge in potential_edges:
                if edges[1][potential_edge] not in all_positive_segments:
                    cord = np.fromstring(edges[2][potential_edge][2:-1], dtype=int, sep=' ')
                    cord = [int(cord[0] * resolution_scale+250), int(cord[1] * resolution_scale + 250), int(cord[2] * resolution_scale + 250)]
                    row = pd.DataFrame([{'node0_segid': int(edges[0][potential_edge]), 'node1_segid': int(edges[1][potential_edge]), 'cord':cord, 'target': 0, 'prediction': -1}])
                    row.to_csv(sample_path, mode='a', header=False, index=False)
            continue
        potentials = list(edges[1][np.where(edges[0] == pos[i])[0]])
        if query[i] in potentials:
            hit = hit + 1
            row = pd.DataFrame([{'node0_segid': int(query[i]), 'node1_segid': int(pos[i]), 'hit': 1}])
            row.to_csv(hit_csv_path, mode='a', header=False, index=False)
            # get positive and negative samples
            sample_path = os.path.join(sample_dir, str(pos[i]) + '_' + str(query[i]) + '.csv')
            potential_edges = list(np.where(edges[0] == query[i])[0]) + list(np.where(edges[0] == pos[i])[0])
            idx = np.where(edges[0] == pos[i])[0][np.where(query[i]==potentials)[0][0]]
            cord = np.fromstring(edges[2][idx][2:-1], dtype=int, sep=' ')
            cord = [int(cord[0]*resolution_scale+250), int(cord[1]*resolution_scale+250), int(cord[2]*resolution_scale+250)]
            row = pd.DataFrame([{'node0_segid': int(pos[i]), 'node1_segid': int(query[i]), 'cord':cord, 'target': 1, 'prediction': -1}])
            row.to_csv(sample_path, mode='a', header=False, index=False)
            for potential_edge in potential_edges:
                if edges[1][potential_edge] not in all_positive_segments:
                    cord = np.fromstring(edges[2][potential_edge][2:-1], dtype=int, sep=' ')
                    cord = [int(cord[0] * resolution_scale+250), int(cord[1] * resolution_scale+250), int(cord[2] * resolution_scale+250)]
                    row = pd.DataFrame([{'node0_segid': int(edges[0][potential_edge]), 'node1_segid': int(edges[1][potential_edge]), 'cord':cord, 'target': 0, 'prediction': -1}])
                    row.to_csv(sample_path, mode='a', header=False, index=False)
            continue
        row = pd.DataFrame([{'node0_segid': int(query[i]), 'node1_segid': int(pos[i]), 'hit': 0}])
        row.to_csv(hit_csv_path, mode='a', header=False, index=False)
    print('bilogical edge recall: ', hit / total_positives)


def test_biological(label_name, unknown_filename_h5, volume):
    mapping = readh5(label_name, 'original')
    edges = readh5(unknown_filename_h5)
    csv_path = label_name.replace('.h5', '_result.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

    for edge in tqdm(edges):
        seg_start = mapping[edge[3] - 1]
        seg_candidate = mapping[edge[4] - 1]
        # cord = np.asarray([edge[0], edge[1], edge[2]]) - np.asarray([8, 64, 64])
        # upper_bound = np.asarray([volume.shape[2], volume.shape[1], volume.shape[0]]) - np.asarray([16, 128, 128])
        # cord = [np.clip(cord[0], 0, upper_bound[0]), np.clip(cord[1], 0, upper_bound[1]), np.clip(cord[2], 0, upper_bound[2])]
        cord = np.asarray([edge[0], edge[1], edge[2]])
        row = pd.DataFrame(
            [{'node0_segid': int(seg_start), 'node1_segid': int(seg_candidate), 'cord': cord, 'target': -1,
              'prediction': -1}])
        row.to_csv(csv_path, mode='a', header=False, index=False)
    stat_biological_recall(csv_path, '/braindat/lab/wangcx/xray-challenge-eval-master/p_val.csv')



# the prefix name corresponds to the meta file in meta/{PREFIX}.meta

prefix = 'xray-test-downsample'
print(prefix)

# read the input segmentation data
# note that grid size should be the same with segmentation shape in out application
segmentation = dataIO.ReadSegmentationData(prefix)
segmentation = np.transpose(segmentation, [2,1,0])

# subset is either training, validation, or testing
subset = 'testing'

# remove the singleton slices
# node_generation.RemoveSingletons(prefix, segmentation)

# need to update the prefix and segmentation
# removesingletons writes a new h5 file to disk
# prefix = '{}-segmentation-wos'.format(prefix)
# segmentation = dataIO.ReadSegmentationData(prefix)
# need to rerun seg2gold mapping since segmentation changed
# seg2gold_mapping = seg2gold.Mapping(prefix, segmentation, gold)


# generate locations for segments that are too small
# node_generation.GenerateNodes(prefix, segmentation, subset, seg2gold_mapping)

# run inference for node network
# node_model_prefix = 'architectures/nodes-400nm-3x20x60x60-Kasthuri/nodes'
# nodes.forward.Forward(prefix, node_model_prefix, segmentation, subset, seg2gold_mapping, evaluate=True)

# need to update the prefix and segmentation
# node generation writes a new h5 file to disk
# prefix = '{}-reduced-{}'.format(prefix, node_model_prefix.split('/')[1])
# segmentation = dataIO.ReadSegmentationData(prefix)
# need to rerun seg2gold mapping since segmentation changed
# seg2gold_mapping = seg2gold.Mapping(prefix, segmentation, gold)


# generate the skeleton by getting high->low resolution mappings
# prevent OOM
segmentation, mapping = dataIO.relabel(segmentation)
hist = dataIO.get_seg_volume(segmentation)
dust_thresh = 3000
dust = list(np.where(hist<dust_thresh)[0])

dataIO.WriteH5File(mapping['original'],'/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/'+prefix+'mapping.h5', 'original', compression=True)
#
# # and running topological thinnings
skeleton_resolution = (80, 80, 80)
# seg2seg.DownsampleMapping(prefix, segmentation, output_resolution=skeleton_resolution)
# generate_skeletons.TopologicalThinning(prefix, segmentation, skeleton_resolution=skeleton_resolution)
# generate_skeletons.FindEndpointVectors(prefix, skeleton_resolution=skeleton_resolution)

# run edge generation function
unknown_filename_h5 = edge_generation.GenerateEdges_test(prefix, segmentation, subset, width=(128, 128, 128), dust=dust, skeleton_resolution=skeleton_resolution)
test_biological('/braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs/neuronseg/features/biological/'+prefix+'mapping.h5', unknown_filename_h5, segmentation)


# run inference for edge network
# edge_model_prefix = 'architectures/edges-600nm-3x18x52x52-Kasthuri/edges'
# edges.forward.Forward(prefix, edge_model_prefix, subset)
#
# # run lifted multicut
# lifted_multicut.LiftedMulticut(prefix, segmentation, edge_model_prefix)
