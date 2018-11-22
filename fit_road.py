# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:55:12 2018

@author: avarfolomeev
"""

def fit_road(data_):

    data = data_.copy()
    avg = np.mean(data,0)
    for i in range(data.shape[0]):
        data[i] = data[i] - avg

    #fit Z against X-Y  plane!!!
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    return C, avg


def get_road_rotation(cloud):

    #limits = np.array([[0.1, 30], [-10,10], [-2.2,2]])
    
    X,Y = np.meshgrid(np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5))
   # C,avg = fit_road(filter_cloud(cloud, limits)[0])
    C,avg = fit_road(cloud)
    Z = C[0]*X + C[1]*Y + C[2]

    
    p0 = np.array([X[0,0],Y[0,0],Z[0,0]])
    p1 = np.array([X[-1,0],Y[-1,0],Z[-1,0]])
    p2 = np.array([X[-1,-1],Y[-1,-1],Z[-1,-1]])
    p3 = np.array([X[0,-1],Y[0,-1],Z[0,-1]])
    v0 = p3-p0
    v1 = p1-p0
    nrm = np.cross(v0,v1)
    nrm = nrm/np.linalg.norm(nrm)
    #print(v0,v1,nrm)
    
    target = np.array([0,0,1]); 
    rot = find_rotation(nrm, target);
    return avg, rot    


#def remove_yaw(rm):
#    eul = rotationMatrixToEulerAngles(rm); #xyz in Z-Y-X (ext) order
#    angle = [0,0, 0-eul[2]];#

#    return ( eulerAnglesToRotationMatrix(angle) * rm);


# Inputs:
#  road_to_lidar - what I've measured on the road
#  imu -        - IMU rotation for that frame      
def get_imu_to_lidar_rotation(world_to_lidar, world_to_imu):
    world_to_lidar_no_yaw = remove_yaw(world_to_lidar);
    world_to_imu_no_yaw = remove_yaw(world_to_imu);
    #imu_to_lidar = (world_to_imu_no_yaw.transpose() * world_to_lidar_no_yaw )
    imu_to_lidar = (world_to_imu.transpose() * world_to_lidar )
    return imu_to_lidar;
    
#%%

def plot_and_fit_road(ax, plane, color='b'):
    scatt3d(ax,plane,False,color)
    X,Y = np.meshgrid(np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5))
   # C,avg = fit_road(filter_cloud(cloud, limits)[0])
    C,avg = fit_road(plane)
    Z = C[0]*X + C[1]*Y + C[2]

    ax.plot_surface(X+avg[0], Y+avg[1], Z+avg[2], rstride=1, cstride=1, alpha=0.2,
                   color=color)
    cx = C; #np.array([C[2],C[0],C[1]])
    c = cx /np.linalg.norm(cx)
    p0 = np.array([X[0,0],Y[0,0],Z[0,0]])
    p1 = np.array([X[-1,0],Y[-1,0],Z[-1,0]])
    p3 = np.array([X[0,-1],Y[0,-1],Z[0,-1]])
    p2 = np.array([X[-1,-1],Y[-1,-1],Z[-1,-1]])
    plot_line(ax,p0+avg,c,color,'o')
    plot_line(ax,p1+avg,c,color,'s')
    plot_line(ax,p2+avg,c,color,'*')
    plot_line(ax,p3+avg,c,color,'d')
    v0 = p2-p0
    v1 = p3-p1
    nrm = np.cross(v0,v1)
    nrm = nrm/np.linalg.norm(nrm)
    plot_line(ax,avg,c,color,'o')
    plot_line(ax,avg,nrm,color,'*')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax_3d.set_zlim(-1,1)
    ax.axis('equal')

    return c, avg, np.array([p0,p1,p2,p3])
    
 #%%
plane_dir = 'E:\\Data\\Voxels\\201809_usa\\planes\\test14_4\\head'
road_clouds = [];
for i in [1,2,3,4,5,6,9,10]:
    road_clouds.append(read_cloud_csv(i, plane_dir)[0])
    
#%%



road2lidar=[]
for i in range(len(road_clouds)):
    road2lidar = get_road_rotation(road_clouds[i])[1]

world2imu_q = [Quaternion(matrix = r) for r in world2imu]
mean_q = sum(world2imu_q/8)
world2road = mean_q.rotation_matrix #error!!!!
road2imu = mean_q.rotation_matrix #[  -0.94101673,    2.32569893,  130.57125182]
imu2lidar=[r*remove_yaw(road2imu).transpose() for r in road2lidar]

imu2lidar_q = [Quaternion(matrix = r) for r in imu2lidar]        
imu2lidar_mean = np.matrix((sum(imu2lidar_q)/len(imu2lidar_q)).rotation_matrix)
               
               
clouds_rot =  [road_clouds[i]*imu2lidar[i].transpose()*world2imu[i].transpose() for i in range(8)]
clouds_rot1 =  [road_clouds[i]*imu2lidar_mean.transpose()*world2imu[i].transpose() for i in range(8)]               

   