# ACNP
This is the code of our paper 'Enhancing Octree-based Context Models for Point Cloud Geometry Compression with Attention-based Child Node Number Prediction'. It can be used for object point clouds and LiDAR point clouds compression.
# Requirements
python 3.7
PyTorch 1.9.0+cu102
file/environment.sh to help you build this environment
# Train
First, it is necessary to completely disable the class 'model' in the file ACNPoctAttention.py and only train 'model2'. Then, disable the backpropagation of 'model2', retaining only the backpropagation of 'model', and subsequently train 'model'.

python EMRoctAttention.py 
# Test
python test.py 
