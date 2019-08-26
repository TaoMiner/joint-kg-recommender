import visdom
import time
import numpy as np
 
class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
    
    def log(self, output_str, win_name="Log"):
        x = self.index.get(win_name, 0)
        self.vis.text(output_str, win=win_name, append=False if x == 0 else True)
        self.index[win_name] = x + 1

    def plot_many_stack(self, points, win_name="", options={}):
        '''
        self.plot('loss',1.00)
        '''
        name=list(points.keys())
        if len(win_name) < 1:
            win_name = " ".join(name)

        options['legend'] = name
        options['title'] = win_name

        x = self.index.get(win_name, 0)
        val=list(points.values())
        if len(val)==1:
            y=np.array(val)
        else:
            y=np.array(val).reshape(-1,len(val))
        
        self.vis.line(Y=y,X=np.ones(y.shape)*x,
                    win=win_name,
                    opts=options,
                    update=None if x == 0 else 'append'
                    )
        self.index[win_name] = x + 1            
