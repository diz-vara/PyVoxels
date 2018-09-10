# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:42:11 2018

@author: avarfolomeev
"""

cmap = ['blue','dodgerblue','cyan', 'lightgreen', 'green', 'yellow', 
        'orange', 'red', 'magenta', 'brown',  'darkgray'];
        
        
draw3d = True
start = 1

a2.cla()

ff = np.arange(start,start+11,1)


filter_h = (ptR1_11a[:,2] > -1.4) & (ptR1_11a[:,2] < 10.1)

if (draw3d):
    a3.cla()


    
for frame in ff:
    filter_frame = (fR1_11a==frame)
    a2.scatter(ptR1_11a[filter_frame & filter_h,0], 
               ptR1_11a[filter_frame & filter_h,1],
               marker='.',
               edgecolors='face',
               color=cmap[(frame-start)%11],s=1)
    a2.scatter(nR1_11a[filter_frame,0], 
               nR1_11a[filter_frame,1],marker='x',
               edgecolors='face',
               color=cmap[(frame-start)%11],s=16)

    if (draw3d):
        a3.scatter(0-ptR1_11a[filter_frame,0], 
                   ptR1_11a[filter_frame,1],
                   ptR1_11a[filter_frame,2],
                   marker='.',edgecolors='face',color=cmap[frame%11],s=1)
    
        a3.scatter(0-nR1_11a[filter_frame,0], 
                   nR1_11a[filter_frame,1],
                   nR1_11a[filter_frame,2],
                   marker='.',edgecolors='face',color=cmap[frame%11],s=3)
    
 #%%
if (False):
 for frame in ff:
    filter_frame = (fR1_11==frame)
    a3.scatter(ptR1_11[filter_range & filter_frame,0], 
               ptR1_11[filter_range & filter_frame,1],
               ptR1_11[filter_range & filter_frame,2],
               marker='.',edgecolors='face',color=cmap[frame%11],s=5)       
    
 #%%

def plot_vector(ax, enu, q, num):
    x0 = enu[num][0]
    y0 = enu[num][1]

    x = np.matrix([20,0,0])
    p1 = (x*np.matrix(q[num].rotation_matrix))[0]
    x1 = x0 + p1[0,0]
    y1 = y0 + p1[0,1]
    ax.plot([x0,x1], [y0,y1],color='r')
    ax.scatter(x1, y1,color='r',marker='.')
    ax.scatter(x0, y0,color='m',marker='.')
    
    
#%%
def plot_orientations(ax, enu, q, color='b', step=300):
    ax.scatter(enu[:,0], 
               enu[:,1],
               marker='.',
               color=color,s=1)    
    for i in range(0,len(enu),step):
        plot_vector(ax,enu,q,i)