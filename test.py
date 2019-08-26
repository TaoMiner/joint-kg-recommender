from jTransUP.utils.visuliazer import Visualizer
import random
import visdom


vis = Visualizer()
for i in range(100):
    x = random.random()
    vis.plot_many_stack({'x': x}, win_name="train_loss", options={'height':10,'width':10})
    y = random.random()
    z = random.random()
    vis.plot_many_stack({'y': y, 'z':z}, win_name="valid_loss", options={'height':300,'width':400})
'''

vis = visdom.Visdom()

trace = dict(x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')
layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})
'''