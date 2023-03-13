import numpy as np
import matplotlib.pyplot as plt

obs = np.load('/home/haochen/TPDM_transformer/obs_test.npy')

obs = np.reshape(obs, (-1,8,6,4))
obs = np.transpose(obs,[0,2,1,3])
plt.figure()

for i in range(6):
    enc = obs[16,i]
    x,y = enc[:,0],enc[:,1]
    print(i)
    print(x)
    print(y)
    plt.plot(x,y)
    plt.scatter(x[-1], y[-1],marker='*')
plt.legend(['ego']+['nei'+str(i+1) for i in range(5)])
plt.savefig('/home/haochen/TPDM_transformer/test_traj.png')   