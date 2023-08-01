cd /braindat/lab/liusl/flywire/biologicalgraphs/biologicalgraphs
cd algorithms
python setup.py build_ext --inplace
cd ../evaluation
python setup.py build_ext --inplace
cd ../graphs/biological
python setup.py build_ext --inplace
cd ../../skeletonization
python setup.py build_ext --inplace
cd ../transforms
python setup.py build_ext --inplace
pip install numba