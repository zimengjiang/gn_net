import torch
import numpy as np  

def check_batch_slice():
    a = np.array([[1,2],[3,4]])
    a = torch.from_numpy(a)
    b = a + 4
    c = a * (-1)
    d = torch.stack((a,b,c,),dim = 0)
    f = torch.stack((d,-d),dim=0)
    idx = np.random.randint(2, size = (2,8,2))

    b, c, h, w = f.shape
    N = idx.shape[1]
    
    f_permuted = f.permute(1,0,2,3)
    f_2d = f_permuted.reshape((c, b*h*w))
    f_idx_2d = np.zeros((b*N))

    for b_th in range(b):
            m = idx[b_th]
            f_idx_2d[b_th*(N):(b_th+1)*N] = b_th*w*h + m[:,1]*w + m[:,0]
    f_idx_2d = torch.from_numpy(f_idx_2d).type(torch.LongTensor)
    f_ext = torch.index_select(f_2d, -1, f_idx_2d).transpose(0,1)
    tmp
    


if __name__ == "__main__":
    check_batch_slice()