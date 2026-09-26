# How to open tensorboard
Note the way "/" is used; a file path copied from the right-click menu can be used directly
tensorboard --logdir="<AUTHOR_DATA_ROOT>/MONAILabel/AD/3D_AD_densenet_WeightedCrossEntropyLoss/run"




import matplotlib.pyplot as plt
# Set a Chinese font (prefer a system Chinese font)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# If problems persist after the settings above, you can force the default matplotlib font
plt.rcParams['font.family'] = 'sans-serif'

