from conf import *
from main import *


save_dir = Path('results/CartPole-v1/2021-04-25T19-22-14')
class Create_vid():
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.chk_dir = self.save_dir / 'chk'
        self.res_arr_dir = self.save_dir / 'res_arrays' 
        self.res_arr_pths = list(self.res_arr_dir.rglob('*.npy'))

        self.res_arr = {}
        for pth in self.res_arr_pths:
            seed = pth.name.split('_')[0]
            print(type(dict(np.load(pth, allow_pickle = True))))
            self.res_arr[seed] = np.load(pth, allow_pickle = True)[0]['reward']
            self.res_arr[seed+'_total'] = sum(self.res_arr[seed])  
        



vid = Create_vid(save_dir)


