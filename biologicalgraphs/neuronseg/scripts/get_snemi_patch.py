import sys

from biologicalgraphs.utilities import dataIO
from biologicalgraphs.graphs.biological import edge_generation_get_patch
from biologicalgraphs.transforms import seg2seg, seg2gold
from biologicalgraphs.skeletonization import generate_skeletons
from skimage.transform import resize

# amap = {"A":"_BC_A", "B":"_AC_B", "C":"_AB_C"}
# akey = "A"
#
# for akey in["A", "B", "C"]:
#     prefix = 'cremi-sample{}-segmentation-wos-reduced-nodes-400nm-3x20x60x60-cremi{}-finetune'.format(akey, amap[akey])
#     #prefix = 'cremi-sample{}-segmentation-wos'.format(akey)
#
#     # prefix = 'cremi-sample{}'.format(akey)
#     print("prefix", prefix)
#
#     gold = dataIO.ReadGoldData(prefix)
#     segmentation = dataIO.ReadSegmentationData(prefix)
#
#     # new mapping
#     seg2gold_mapping = seg2gold.Mapping(prefix, segmentation, gold, use_cache=False)
#
#
#     seg2seg.DownsampleMapping(prefix, segmentation)
#     generate_skeletons.TopologicalThinning(prefix, segmentation)
#     generate_skeletons.FindEndpointVectors(prefix)
#
#
#     # generate edges
#     subset = 'testing'
#     edge_generation.GenerateEdges(prefix, segmentation, subset, seg2gold_mapping)


# FOR SNEMI3D
subsets = ['training', 'testing']
prefixes = ['SNEMI3D-train', 'SNEMI3D-test']
for subset, prefix in zip(subsets, prefixes):
    print("prefix: {},  subset: {}".format(prefix, subset))
    # SNEMI3D-test-segmentation-wos-reduced-nodes-400nm-3x20x60x60-SNEMI3D.h5

    prefix = '{}-segmentation-wos-reduced-nodes-0-17x129x129-SNEMI3D'.format(prefix)
    print("prefix", prefix)

    gold = dataIO.ReadGoldData(prefix)
    segmentation = dataIO.ReadSegmentationData(prefix)
    image = dataIO.ReadImageData(prefix)

    target_shape = [100, 512, 512]
    image = resize(image, target_shape, order=1,
                          mode='constant', cval=0, clip=True, preserve_range=True,
                          anti_aliasing=True)
    gold = resize(gold, target_shape, order=0,
                          mode='constant', cval=0, clip=True, preserve_range=True,
                          anti_aliasing=False)
    segmentation = resize(segmentation, target_shape, order=0,
                          mode='constant', cval=0, clip=True, preserve_range=True,
                          anti_aliasing=False)

    # new mapping
    seg2gold_mapping = seg2gold.Mapping(prefix, segmentation, gold)

    seg2seg.DownsampleMapping(prefix, segmentation)
    generate_skeletons.TopologicalThinning(prefix, segmentation)
    generate_skeletons.FindEndpointVectors(prefix)

    # generate edges
    edge_generation_get_patch.GenerateEdges(prefix, segmentation, subset, seg2gold_mapping, gold, image)

    # training
    # No.Positive
    # Edges: 629
    # No.Negative
    # Edges: 3963
    # No.Unknown
    # Edges: 206

    # testing   tesing生成后，复制一份到validataion中
    # No.Positive
    # Edges: 469
    # No.Negative
    # Edges: 3273
    # No.Unknown
    # Edges: 471

    # 对比一下Kasthuri   比例是差不多的
    # No.Positive
    # Edges: 1802
    # No.Negative
    # Edges: 13685

print("DONE")


