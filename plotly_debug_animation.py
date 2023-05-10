import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
            }



def plotly_intermediate_goals(achieved_init_states,achieved_trajectories,desired_goals,initial_goals,pool_goals_container,epoch,cycle):
    fig = plotly.subplots.make_subplots(rows=1, cols=1, subplot_titles=[f"Diffusion-model"],specs=[[{"type": "scatter3d"}]])

    fig.add_trace(go.Scatter3d(x=initial_goals[:, 0], y=initial_goals[:, 1], z=initial_goals[:, 2], name='init',mode='markers', marker=dict(size=8,color='blue',opacity=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=desired_goals[:, 0], y=desired_goals[:, 1], z=desired_goals[:, 2], name='init',mode='markers', marker=dict(size=8,color='red',opacity=0.8)), row=1, col=1)
    pool_goals = pool_goals_container[0]
    fig.add_trace(go.Scatter3d(x=pool_goals[:, 0], y=pool_goals[:, 1], z=pool_goals[:, 2], name='pool',mode='markers', marker=dict(size=8,color=plotly.colors.DEFAULT_PLOTLY_COLORS[0],opacity=0.8)), row=1, col=1)
    # fig.add_trace(go.Scatter3d(x=achieved_init_states[:, 0], y=achieved_init_states[:, 1], z=achieved_init_states[:, 2], name='ac_in',mode='markers', marker=dict(size=8,color='pink',opacity=0.8)), row=1, col=1)
    # pool_goals = pool_goals_container[:,0]
    # fig.add_trace(go.Scatter3d(x=pool_goals[:, 0], y=pool_goals[:, 1], z=pool_goals[:, 2], name='pool',mode='markers', marker=dict(size=8,color=plotly.colors.DEFAULT_PLOTLY_COLORS[0],opacity=0.8)), row=1, col=1)
    # for i,pool_goals in enumerate(pool_goals_container):
    #     fig.add_trace(go.Scatter3d(x=pool_goals[:, 0], y=pool_goals[:, 1], z=pool_goals[:, 2], name='pool',mode='markers', marker=dict(size=8,color=plotly.colors.DEFAULT_PLOTLY_COLORS[i],opacity=0.8)), row=1, col=1)

    # Frames
    frames = [go.Frame(data= [go.Scatter3d(x=pool_goals[:,0],
                                        y=pool_goals[:,1],
                                        z=pool_goals[:,2]
                                        )
                            ],
                    traces= [2],
                    name=f'frame{k}'      
                    )for k,pool_goals in  enumerate(pool_goals_container)]
    fig.update(frames=frames)

    # for i in range(achieved_trajectories.shape[0]):
    #     fig.add_trace(go.Scatter3d(x=[achieved_trajectories[i, 0, 0]], y=[achieved_trajectories[i, 0, 1]], z=[achieved_trajectories[i, 0, 2]] ,name='ac_tj', mode='markers', marker=dict(size=12,color='yellow',opacity=0.8)), row=1, col=1)

    #     fig.add_trace(go.Scatter3d(x=achieved_trajectories[i, 1:, 0], y=achieved_trajectories[i, 1:, 1], z=achieved_trajectories[i,1:, 2], name='ac_tj', line=dict(color='black', width=4)), row=1, col=1)
    
    sliders = [
    {"pad": {"b": 10, "t": 60},
     "len": 0.9,
     "x": 0.1,
     "y": 0,
     
     "steps": [
                 {"args": [[f.name], frame_args(0)],
                  "label": str(k),
                  "method": "animate",
                  } for k, f in enumerate(fig.frames)
              ]
     }
        ]

    fig.update_layout(

        updatemenus = [{"buttons":[
                        {
                            "args": [None, frame_args(50)],
                            "label": "Play", 
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "Pause", 
                            "method": "animate",
                    }],
                        
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
        )

    fig.update_layout(scene=dict(
    xaxis=dict(nticks=4, range=[0, 1.5], ),
    yaxis=dict(nticks=4, range=[0, 1.5], ),
    zaxis=dict(nticks=4, range=[0, 1.5], ),
    aspectratio=dict(x=1, y=1, z=1),
    aspectmode='manual'
    ))

    fig.update_layout(sliders=sliders)
    # fig.show()
    fig.write_html("container/hgg_debug"+str(epoch) + str(cycle) + ".html")


epoch=0
cycle=1
pool_goals_container = []
achieved_init_states = np.load(f"container/achieved_init_states_{epoch}_cycle{cycle}.npy")
achieved_trajectories = np.load(f"container/achieved_trajectories_{epoch}_cycle{cycle}.npy")
desired_goals = np.load(f"container/desired_goals_ep_{epoch}_cycle{cycle}.npy")
initial_goals = np.load(f"container/initial_goals_ep_{epoch}_cycle{cycle}.npy")

for epoch in range(0,15):
    for cycle in range(3,20,1):
        
        pool_goals = np.load(f"container/pool_goals_ep_{epoch}_cycle{cycle}.npy")
        # pool_goals = np.load(f"container/diffusion_model/d_goal{epoch}_cycle_{cycle}.npy")
        pool_goals_container.append(pool_goals)



plotly_intermediate_goals(achieved_init_states,achieved_trajectories,desired_goals,initial_goals,pool_goals_container,epoch,cycle)