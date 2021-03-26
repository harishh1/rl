
from imports import *

class ReplayBuffer():
    def __init__(self, max_size = 10000, batch_size = 64):
        self.as_mem = np.empty(shape=(max_size), dtype = np.ndarray) #action
        self.ss_mem = np.empty(shape=(max_size), dtype = np.ndarray) #state memory
        self.rs_mem = np.empty(shape=(max_size), dtype = np.ndarray) #reward
        self.ps_mem = np.empty(shape=(max_size), dtype = np.ndarray) #?
        self.ds_mem = np.empty(shape=(max_size), dtype = np.ndarray) # done flag
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def store(self, sample):
        s, a, r, p, d = sample
        self.as_mem[self._idx] = a 
        self.ss_mem[self._idx] = s 
        self.rs_mem[self._idx] = r 
        self.ps_mem[self._idx] = p 
        self.ds_mem[self._idx] = d         

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size = None):
        if batch_size == None: batch_size = self.batch_size

        idxs = np.random.choice(self.size, batch_size, replace = False)
        experiences = np.vstack(self.ss_mem[idxs]), \
        np.vstack(self.as_mem[idxs]), \
        np.vstack(self.rs_mem[idxs]), \
        np.vstack(self.ps_mem[idxs]), \
        np.vstack(self.ds_mem[idxs])

        return experiences

    def __len__(self):
        return self.size

def get_videos_html(env_videos, title, max_n_videos=5):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        video = io.open(video_path, 'r+b').read()
        encoded = base64.b64encode(video)

        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <video width="960" height="540" controls>
            <source src="data:video/mp4;base64,{1}" type="video/mp4" />
        </video>"""
        strm += html_tag.format('Episode ' + str(meta['episode_id']), encoded.decode('ascii'))
    return strm


def get_gif_html(env_videos, title, subtitle_eps=None, max_n_videos=4):
    videos = np.array(env_videos)
    if len(videos) == 0:
        return
    
    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1,]
    videos = videos[idxs,...]

    strm = '<h2>{}<h2>'.format(title)
    for video_path, meta_path in videos:
        basename = os.path.splitext(video_path)[0]
        gif_path = basename + '.gif'
        if not os.path.exists(gif_path):
            ps = subprocess.Popen(
                ('ffmpeg', 
                 '-i', video_path, 
                 '-r', '7',
                 '-f', 'image2pipe', 
                 '-vcodec', 'ppm',
                 '-crf', '20',
                 '-vf', 'scale=512:-1',
                 '-'), 
                stdout=subprocess.PIPE)
            output = subprocess.check_output(
                ('convert',
                 '-coalesce',
                 '-delay', '7',
                 '-loop', '0',
                 '-fuzz', '2%',
                 '+dither',
                 '-deconstruct',
                 '-layers', 'Optimize',
                 '-', gif_path), 
                stdin=ps.stdout)
            ps.wait()

        gif = io.open(gif_path, 'r+b').read()
        encoded = base64.b64encode(gif)
            
        with open(meta_path) as data_file:    
            meta = json.load(data_file)

        html_tag = """
        <h3>{0}<h3/>
        <img src="data:image/gif;base64,{1}" />"""
        prefix = 'Trial ' if subtitle_eps is None else 'Episode '
        sufix = str(meta['episode_id'] if subtitle_eps is None \
                    else subtitle_eps[meta['episode_id']])
        strm += html_tag.format(prefix + sufix, encoded.decode('ascii'))
    return strm

def create_res_dir(current_dir,conf_json, from_pixel):
    res = current_dir.joinpath('results',) 
    res.mkdir(parents=True, exist_ok=True)#create res dir
    list_dir = [x for x in res.iterdir() if x.is_dir()] #list dir

    max_dir_val = max([int(m.name.split('_')[1]) for m in list_dir] or [0]) #max val of dir

    dir_name = ('pixel' if from_pixel else 'value') + '_' + str(max_dir_val+1)
    d = res.joinpath(dir_name)
    d.mkdir()
    cp_dir = d.joinpath('checkpoints')
    cp_dir.mkdir(parents=True, exist_ok=True)

    with open(str(d.joinpath(dir_name+'_conf.json')), 'w', encoding='utf-8') as f:
        f.write(conf_json)

    return d,cp_dir




