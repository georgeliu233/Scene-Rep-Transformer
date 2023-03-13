import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


pca = PCA(n_components=2)

x,y = np.array(data[0][:300]) , np.array(data[1][:300])
print(x.shape,y.shape)
pca.fit(x)

new_x = pca.transform(x)
min_new = np.min(new_x,axis=0)
max_new = np.max(new_x,axis=0)
new_x = (new_x - min_new) / (max_new - min_new)
print(new_x.shape)

inds = np.argsort(y[:,0])
mask = y[:,0] < 1.2
percentile = int(0.25*len(inds))
low, high = new_x[inds[:percentile],:] , new_x[inds[-percentile:],:]
print(np.linalg.norm(np.mean(low,axis=0) - np.mean(high,axis=0)))
print(np.max(y[mask, 0]), np.min(y[mask, 0]))
print(new_x.shape)

fig = plt.figure()
sns.set(style="whitegrid", font_scale=2, rc={"lines.linewidth": 3.5})
sc = plt.scatter(new_x[mask, 0], new_x[mask, 1], c=y[mask, 0],marker='o',s=35,cmap='Blues',linewidths=0.5,edgecolors='k')
plt.colorbar(sc)  
plt.savefig('./cross_pca_cl.png')

