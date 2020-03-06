# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:36:55 2020

@author: avarfolomeev
"""

rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,121,2))
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,121,5))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,242,5))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(50,200))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(50,150))
rot,t = _proc_cal_idx([7,6], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(50,150))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,130))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,121))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,122))
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',1)
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #if (rotate is not None):
    #    sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,121))
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',1)
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #if (rotate is not None):
    #    sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return _cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,121))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,122))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,5,242))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5, tvec=t, rvec=rot, points_idx = np.arange(0,242))
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',1)
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    if (rotate is not None):
        sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    pts_idx = np.arange(121);                       
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx]
            boards = boards[points_idx]
        
        ret, rot, t, inl = cv2.solvePnPRansac(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',1)
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #if (rotate is not None):
    #    sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return _cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
eulerAnglesToRotationMatrix([-1.1413807   1.60789584  1.48490147])
eulerAnglesToRotationMatrix([-1.1413807,1.60789584,1.48490147])
rm_to_degrees(eulerAnglesToRotationMatrix([-1.1413807,1.60789584,1.48490147]))
rm_to_degrees(eulerAnglesToRotationMatrix(rot))
rot67,t67 = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
rot67,t67 = _proc_cal_idx([6,7], rotate = e, VLP32_multi=1., scale=0.5)
rot67,t67 = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
rot7,t7 = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5,points_idx=[0,11,111,61,121])
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5,points_idx=[0,10,110,60,120])
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    pts_idx = np.arange(121);                       
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t, inl = cv2.solvePnPRansac(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5,points_idx=[0,10,110,60,120])

clouds
points_idx=[0,10,110,60,120]


cl
cl=clouds[points_idx,:]
bd = boards[points_idx,:]
cv2.solvePnP(cl,bd,mtx,dist)
cv2.solvePnPRansac(cl,bd,mtx,dist)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    pts_idx = np.arange(121);                       
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t, inl = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5,points_idx=[0,10,110,60,120])
rot6,t6,_,_ = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5,points_idx=[0,10,110,60,120])
rot6,t6,_ = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5,points_idx=[0,10,110,60,120])
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    pts_idx = np.arange(121);                       
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5,points_idx=[0,10,110,60,120])
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
def calc_image_grid(num, base_dir,back,ax=None):
    fname = base_dir + '/{:06}.png'.format(num)
    print(fname);
    return load_draw_2d_board(fname,back,ax)



def load_draw_2d_board(name,  mtx, dist, back,ax=None, shape = None,
                       scale = 1.):
    img = cv2.imread(name,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return draw_2d_board(img, mtx, dist, back, ax, shape, scale)


def draw_2d_board(img, mtx, dist, back=False, ax=None, shape = None, scale = 1.):
    #if (not ax is None):
    #    ax.cla()
    
    #uimg=cv2.undistort(img,mtx,dist)
    if ( scale > 0 and scale != 1):
        img = cv2.resize( img, (0,0), fx = scale, fy = scale)
    
    
    if (not shape is None):
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        
        flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
        ret,corn_xy = cv2.findChessboardCorners(grey,shape, flagCorners)
        if (ret):
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            #cv2.cornerSubPix(grey,corn_xy,shape, (-1,-1),term)
        else:
            return
        cxy=np.array(corn_xy[:,0,:]).astype(np.float64)
    
    else:
        shape = (11,11) #LA recordings with manual corners
        corners = ( (img[:,:,0]<=0) & (img[:,:,1]<=0) & (img[:,:,2]> 254) )
        corn_xy=np.nonzero(corners.transpose())
        cxy=np.array(corn_xy).transpose().astype(np.float64)
    
    # for resized (enlarged) imnages!!!
    if (False and img.shape[0] > 1100):
        cxy = cxy / 2;
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    if (not ax is None):
        ax.imshow(img)
    
    
    _cxy = cxy.copy()    
    
    cxy_u = cv2.undistortPoints(np.array([_cxy]),mtx,dist, R=mtx)[0]
    
    
    #if (not ax is None):
    #    scatt2d(ax,cxy_u, False, None, 'o',50)
    
    top_idx = np.argmin(cxy_u[:,1])
    right_idx = np.argmax(cxy_u[:,0])
    
    top = cxy_u[top_idx]
    cxy_ut = cxy_u-top;
    
    right = cxy_ut[right_idx]
    angle = np.degrees(np.arcsin(right[1]/np.linalg.norm(right)))
    rm = np.matrix(cv2.getRotationMatrix2D((0,0), -angle,1)[:,:2])
    
    
    cxy_rot = np.array(cxy_ut * rm);
    
    order = np.argsort(cxy_rot[:,1])
    
    final_order = np.array([]);
    for i in np.arange(0,shape[0]*shape[1],shape[0]):
        row = cxy_rot[order[i:i+shape[1]]]
        row_order = np.argsort(row[:,0])
        final_order = np.append(final_order,order[row_order+i])
    
    
    if (back):    
        final_order = final_order.reshape(shape).transpose().reshape(-1)    
    cxy_ret = cxy[final_order.astype(np.int32)]
    
    pts_idx=[0,10,110,60,120]
    cxy_ret = cxy_ret[pts_idx]
    
    if (not ax is None):
        scatt2d(ax, _cxy, False, None, 'd',30)
        
        for i in range(len(cxy_ret)):
            ax.annotate(str(i+1),np.array(cxy_ret[i]))
    
    return cxy_ret
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
def calc_image_grid(num, base_dir,back,ax=None):
    fname = base_dir + '/{:06}.png'.format(num)
    print(fname);
    return load_draw_2d_board(fname,back,ax)



def load_draw_2d_board(name,  mtx, dist, back,ax=None, shape = None,
                       scale = 1.):
    img = cv2.imread(name,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return draw_2d_board(img, mtx, dist, back, ax, shape, scale)


def draw_2d_board(img, mtx, dist, back=False, ax=None, shape = None, scale = 1.):
    #if (not ax is None):
    #    ax.cla()
    
    #uimg=cv2.undistort(img,mtx,dist)
    if ( scale > 0 and scale != 1):
        img = cv2.resize( img, (0,0), fx = scale, fy = scale)
    
    
    if (not shape is None):
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        
        flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
        ret,corn_xy = cv2.findChessboardCorners(grey,shape, flagCorners)
        if (ret):
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            #cv2.cornerSubPix(grey,corn_xy,shape, (-1,-1),term)
        else:
            return
        cxy=np.array(corn_xy[:,0,:]).astype(np.float64)
    
    else:
        shape = (11,11) #LA recordings with manual corners
        corners = ( (img[:,:,0]<=0) & (img[:,:,1]<=0) & (img[:,:,2]> 254) )
        corn_xy=np.nonzero(corners.transpose())
        cxy=np.array(corn_xy).transpose().astype(np.float64)
    
    # for resized (enlarged) imnages!!!
    if (False and img.shape[0] > 1100):
        cxy = cxy / 2;
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    if (not ax is None):
        ax.imshow(img)
    
    
    _cxy = cxy.copy()    
    
    cxy_u = cv2.undistortPoints(np.array([_cxy]),mtx,dist, R=mtx)[0]
    
    
    #if (not ax is None):
    #    scatt2d(ax,cxy_u, False, None, 'o',50)
    
    top_idx = np.argmin(cxy_u[:,1])
    right_idx = np.argmax(cxy_u[:,0])
    
    top = cxy_u[top_idx]
    cxy_ut = cxy_u-top;
    
    right = cxy_ut[right_idx]
    angle = np.degrees(np.arcsin(right[1]/np.linalg.norm(right)))
    rm = np.matrix(cv2.getRotationMatrix2D((0,0), -angle,1)[:,:2])
    
    
    cxy_rot = np.array(cxy_ut * rm);
    
    order = np.argsort(cxy_rot[:,1])
    
    final_order = np.array([]);
    for i in np.arange(0,shape[0]*shape[1],shape[0]):
        row = cxy_rot[order[i:i+shape[1]]]
        row_order = np.argsort(row[:,0])
        final_order = np.append(final_order,order[row_order+i])
    
    
    if (back):    
        final_order = final_order.reshape(shape).transpose().reshape(-1)    
    cxy_ret = cxy[final_order.astype(np.int32)]
    
    pts_idx=[0,10,110,60,120]
    cxy_ret = cxy_ret[pts_idx]
    _cxy = _cxy[pts_idx]
    
    if (not ax is None):
        scatt2d(ax, _cxy, False, None, 'd',30)
        
        for i in range(len(cxy_ret)):
            ax.annotate(str(i+1),np.array(cxy_ret[i]))
    
    return cxy_ret
rot6,t6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
cl
cl=6
_cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                 _grid=(0.0597, 11, 0.211),
                                 delimiter = ',',rotate=rotate,
                                 VLP32_multi = VLP32_multi); 

rotate
rotate=lr
_cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                 _grid=(0.0597, 11, 0.211),
                                 delimiter = ',',rotate=rotate,
                                 VLP32_multi = VLP32_multi);
_cloud
_grid
back
scale
_board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                            scale = 1.);

_board
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = scale);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        #if (scale is not None and scale > 0):
        #    boards = boards * scale
        #else:
        #    scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7,8], rotate = lr, VLP32_multi=1., scale=0.5)
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
rot,t = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7,8], rotate = lr, VLP32_multi=1., scale=0.5)
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
rot,t = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6,7,8], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7,6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7,6,8], rotate = lr, VLP32_multi=1., scale=0.5)
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t, inl = cv2.solvePnPRansac(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot,t = _proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
c3_7_ransac = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
c3_6_ransac = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
c3_8_ransac = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t, inl = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
c3_7c = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
c3_7 = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
c3_7 = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
c3_7_ransac
c3_8 = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
c3_6 = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
c3_6_ransac
colors2labels_o('e:/data/segm/201911_sf/20191227/','color_mask','labels',ont[0])
_proc_cal_idx([1], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([2], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([3], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([4], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([5], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([4,7], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([3,7], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([9], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
_proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            imgU = img.copy()
            mtxU = mtx.copy()
            
            cv2.undistort(img, mtx, dist, imgU, mtxU)
            
            
            _board = load_draw_2d_board(imgU,mtxU, np.zeros(5)), back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
_proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([6,7], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
plt.imshow(img)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            imgU = img.copy()
            mtxU = mtx.copy()
            
            cv2.undistort(img, mtx, dist, imgU, mtxU)
            
            
            _board = load_draw_2d_board(imgU,mtxU, np.zeros(5)), back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            imgU = img.copy()
            mtxU = mtx.copy()
            
            cv2.undistort(img, mtx, dist, imgU, mtxU)
            
            
            _board = load_draw_2d_board(imgU,mtxU, np.zeros(5)), back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, 
                                   rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            imgU = img.copy()
            mtxU = mtx.copy()
            
            cv2.undistort(img, mtx, dist, imgU, mtxU)
            
            
            _board = load_draw_2d_board(imgU,mtxU, np.zeros(5)), back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, rvec, tvec,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            imgU = img.copy()
            mtxU = mtx.copy()
            
            cv2.undistort(img, mtx, dist, imgU, mtxU)
            
            
            _board = load_draw_2d_board(imgU,mtxU, np.zeros(5)), back,ax1,(11,11),
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, rvec, tvec, flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            imgU = img.copy()
            mtxU = mtx.copy()
            
            cv2.undistort(img, mtx, dist, imgU, mtxU)
            
            
            _board = load_draw_2d_board(imgU,mtxU, np.zeros(5)), back,ax1,(11,11), scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, rvec, tvec, flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            imgU = img.copy()
            mtxU = mtx.copy()
            
            cv2.undistort(img, mtx, dist, imgU, mtxU)
            
            
            _board = load_draw_2d_board(imgU,mtxU, np.zeros(5), 
                                        back,ax1,(11,11), 
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, 
                                   rvec, tvec, 
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
_proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
_proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
name = jpglist[cl];
cl
img = cv2.imread(name,-1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgU = img.copy()
mtxU = mtx.copy()

cv2.undistort(img, mtx, dist, imgU, mtxU)

plt.imshow(imgU)
c3_0
a = cv2.undistort(img, mtx, dist)
imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
plt.imshow(img)
ret,mtx,dist = c3_sm_0_0
imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
ret,mtx,dist = c3_sm_0_1
imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
plt.imshow(img)
ret,mtx,dist = c3_sm_0_5
imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
z = np.zeros(5)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
dist = np.zeros(5)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            imgU = img.copy()
            mtxU = mtx.copy()
            
            #imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
            
            
            _board = load_draw_2d_board(img,mtx, np.zeros(5), 
                                        back,ax1,(11,11), 
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, 
                                   rvec, tvec, 
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            #imgU = img.copy()
            #mtxU = mtx.copy()
            
            #imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
            
            
            _board = load_draw_2d_board(img,mtx, np.zeros(5), 
                                        back,ax1,(11,11), 
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, 
                                   rvec, tvec, 
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
_board = load_draw_2d_board(img,mtx, np.zeros(5), 
                            back,ax1,(11,11), 
                            scale = 1.);

name = jpglist[cl];
_board = load_draw_2d_board(img,mtx, np.zeros(5), 
                            back,ax1,(11,11), 
                            scale = 1.);

_board = load_draw_2d_board(img,mtx, dist, 
                            back,ax1,(11,11), 
                            scale = 1.);

_board = load_draw_2d_board(name,mtx, dist, 
                            back,ax1,(11,11), 
                            scale = 1.);

rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
_cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                 _grid=(0.0597, 11, 0.211),
                                 delimiter = ',',rotate=rotate,
                                 VLP32_multi = VLP32_multi); 

def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1.,
                  scale = 1, tvec = None, rvec = None, points_idx = None):
    
    #pts_idx = np.arange(121);                       
    pts_idx=[0,10,110,60,120]
    #pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    
    
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            name = jpglist[cl];
            img = cv2.imread(name,-1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.0597, 11, 0.211),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            
            #imgU = img.copy()
            #mtxU = mtx.copy()
            
            #imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
            
            
            _board = load_draw_2d_board(name,mtx, dist, 
                                        back,ax1,(11,11), 
                                        scale = 1.);
            
            #_cloud = _cloud * rm90
            
            #_grid = _grid[pts_idx]
            #_board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    if (len(clouds) > 0):
        
        if (scale is not None and scale > 0):
            boards = boards * scale
        else:
            scale = 1.
        
        if (points_idx is not None):
            clouds = clouds[points_idx,:]
            boards = boards[points_idx,:]
        
        
        
        ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist, 
                                   rvec, tvec, 
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        
        
        imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
        
        bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
        
        scatt2d(ax1,imgpts[:,0,:]/scale,False,'w','x',size=25)
        scatt2d(ax1,bpts[:,0,:]/scale,False,'b','.',size=1)
        
        sum = cv2.norm(imgpts[:,0,:], boards);
        print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
        print ('reproj error =', sum/len(boards))
    else:
        print ("no good boards")
        rot = t = None
    return rot,t
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6,8], rotate = lr, VLP32_multi=1., scale=0.5)
rot,t = _proc_cal_idx([6,7,8], rotate = lr, VLP32_multi=1., scale=0.5)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, "camera_name":camera}
save_calib_list_json(base_dir + 'Flir_3s_zero_dist.json',[cam_calib_dict])
cam_calib_dict
base_dir
save_calib_list_json(base_dir + 'Flir_3s_zero_dist.json',[cam_calib_dict])
dist = np.zeros((1,5))
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, "camera_name":camera}
save_calib_list_json(base_dir + 'Flir_3s_zero_dist.json',[cam_calib_dict])
img.shape
img_s = cv2.resize(img,(1224,1024))
img_s.shape
img.size()
img.size
imgU = cv2.undistort(img_s, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
plt.imshow(img_s)
gca.clear()
gca().clear()
plt.gca().clear()
plt.imshow(img_s)
plt.imshow(imgU)
plt.imshow(imgU-img_s)
dist
ret,mtx,dist = c3_sm_0_5
imgU = cv2.undistort(img_s, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
plt.imshow(imgU-img_s)
plt.gca().clear()
plt.imshow(imgU)
ret,mtx,dist = c3_sm_0_0
imgU = cv2.undistort(img_s, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
ret,mtx,dist = c3_sm_0_1
plt.imsave("c3_sm_0_0.png",imgU)
imgU = cv2.undistort(img_s, mtx, dist, newCameraMatrix=mtxU)
plt.imsave("c3_sm_0_1.png",imgU)
ret,mtx,dist = c3_sm_0_2
imgU = cv2.undistort(img_s, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
plt.imsave("c3_sm_0_2.png",imgU)
ret,mtx,dist = c3_sm_0_3
imgU = cv2.undistort(img_s, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
plt.imsave("c3_sm_0_3.png",imgU)
ret,mtx,dist = c3_sm_0_4
imgU = cv2.undistort(img_s, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
plt.imsave("c3_sm_0_4.png",imgU)
ret,mtx,dist = c3_sm_0_5
ret
imgU = cv2.undistort(img_s, mtx, dist, newCameraMatrix=mtxU)
plt.imshow(imgU)
plt.imsave("c3_sm_0_5.png",imgU)
plt.imsave("c3_sm_original.png",img_s)
imgU = cv2.undistort(img, mtx, dist, newCameraMatrix=mtxU)
plt.gca().clear()
plt.imshow(imgU)
lidar2imu_spb = eulerAngles2RotationMatrix(np.radians([0.8, -9.2,0]))
lidar2imu_spb = eulerAnglesToRotationMatrix(np.radians([0.8, -9.2,0]))
imu2lidar_spb = np.linalg.inv(lidar2imu_spb)
imu2
save_matrix("imu2lidar_spb.txt", imu2lidar_spb)
np.radians(0.01)
0.0001
sin(0.0001)
sin(0.0001/2)
np.po
sin(np.pi+0.0001/2)
sin(np.pi/6+0.0001/2)
sin(np.pi/6+0.0001/2)-0.5
np.radians(0.05)
np.radians(0.1)
np.arcsin(0.1/30)
np.arcsin(0.1/20)
plt.imshow(gray6)
f3 = plt.figure(3)
plt.imshow(gray6)
f4 = plt.figure(4)
plt.imshow(gray6)
np.sum(gray6==3)
np.sum(gray6==77)
np.sum(gray6==255)
np.sum(gray6==256)
np.count(gray6==256)
a = []
a.append("aa")
a
a.append("aa")
a
sum(gray6==256)
sum(gray6==255)
500%360
(358-2)%360
(2-358)%360
(20-358)%360
(20-18)%360
(18-20)%360
colors2labels_o('e:/data/segm/crossings/2','colors','labels',ont[0])
ont
colors2labels_o('e:/data/segm/crossings/2','colors','labels',ont[0])
ont
colors2labels_o('e:/data/segm/crossings/3','colors','labels',ont[0])
calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=50, sqSide=39.6)
ans
c3_0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=50, sqSide=39.6)
fe = plt.figure()
c3_0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=50, sqSide=39.6)
c3_1 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=50, sqSide=39.6)
cam_3s_n
cam3_0
c3_0
c3_1
c3_2
c3_2 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=50, sqSide=39.6)
c3_3 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=50, sqSide=39.6)
c3_4 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=50, sqSide=39.6)
c3_0
c3_1
c3_2
c3_3
c3_4
c3_5
c3_s0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/s/', model=0, draw_fig = fe, scale = 1.0, nSamples=21, sqSide=39.6)
c3_s0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/s/', model=0, draw_fig = fe, scale = 1.0, sqSide=39.6)
c3_5 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=50, sqSide=39.6)
c3_s0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/s/', model=0, draw_fig = fe, scale = 1.0, sqSide=39.6)
c3_s1 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/s/', model=0, draw_fig = fe, scale = 1.0, sqSide=39.6)
c3_s2 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/s/', model=0, draw_fig = fe, scale = 1.0, sqSide=39.6)
c3_s1 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=20, sqSide=39.6)
c3_s2 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=20, sqSide=39.6)
c3_tp0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=cv2.THIN_PLATE_MODEL, draw_fig = fe, scale = 1.0, nSamples=20, sqSide=39.6)
c3_tp0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/s/', model=cv2.CALIB_THIN_PRISM_MODEL, draw_fig = fe, scale = 1.0, nSamples=20, sqSide=39.6)
c3_tp0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=cv2.CALIB_THIN_PRISM_MODEL, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
c3_tilted0 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=cv2.CALIB_TILTED_MODEL, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
c3_tilted1 = calibrate('E://Data/Voxels/201911_sf/cal_20200203/cam_3/data/', model=cv2.CALIB_TILTED_MODEL, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
c3_tilted1
c3_tp0
c3_s1
c3_s2
c3_s1
c3_1
c3_tilted0
c3_tilted1
c3_1
build_dict(c3_1[1])
build_dict(c3_1[2])
cam_calib_dict = {"mtx":a[1], "dist":a[2], "camera_name":"cam_3"}

a=cam3_1
a=c3_1
save_calib_json("cam_3_5.json", {"mtx":a[1], "dist":a[2], "camera_name":"cam_3"})
save_calib_json("cam_3_5.json", {"mtx":a[1], "dist":a[2], "camera_name":"cam_3", "camera_center":208})
save_calib_json("cam_3_5.json", {"mtx":a[1], "dist":a[2], "camera_name":"cam_3", "camera_center":208,"rot":[], "t":[]})
save_calib_json("cam_3_5.json", {"mtx":a[1], "dist":a[2], "camera_name":"cam_3", "camera_center":208,"rot":[], "t":[]})z=np.array([])
save_calib_json("cam_3_5.json", {"mtx":a[1], "dist":a[2], "camera_name":"cam_3", "camera_center":208,"rot":[], "t":[]})z=np.array([0])
z
z = np.array([])
save_calib_json("cam_3_5.json", {"mtx":a[1], "dist":a[2], "camera_name":"cam_3", "camera_center":208,"rot":z, "t":z})
z = np.zeros((1,1))
save_calib_json("cam_3_5.json", {"mtx":a[1], "dist":a[2], "camera_name":"cam_3", "camera_center":208,"rot":z, "t":z})
a=c3_tilted0
a
a=c3_tilted1
a
save_calib_json("cam_3_tilted.json", {"mtx":a[1], "dist":a[2], "camera_name":"cam_3", "camera_center":208,"rot":z, "t":z})


cam_1_tilted_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data/', model=cv2.CALIB_TILTED_MODEL, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
fe = plt.figure()
cam_1_tilted_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data/', model=cv2.CALIB_TILTED_MODEL, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_1_0_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_1_0_1 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=25, sqSide=39.6)
cam_1_0_2 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=25, sqSide=39.6)
cam_5_0_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_5/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_5_0_1 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_5/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=25, sqSide=39.6)
cam_5_0_2 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_5/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=25, sqSide=39.6)
cam_5_0_0
cam_5_0_1
cam_5_0_2
cam_5_0_3 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_5/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_5_0_4 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_5/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_5_tilt_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_5/data/', model=cv2.CALIB_TILTED_MODEL, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_5_tilt_1 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_5/data/', model=cv2.CALIB_TILTED_MODEL, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_5_tilt_2 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_5/data/', model=cv2.CALIB_TILTED_MODEL, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_5_0_0
cam_5_0_1
cam_5_0_2
cam_5_0_3
cam_5_0_4
cam_5_0n_0 = calibrate('E://Data/Voxels/202002_spb/cal_20200211/cam_5/s/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_5_0n_1 = calibrate('E://Data/Voxels/202002_spb/cal_20200211/cam_5/s/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_5_0n_2 = calibrate('E://Data/Voxels/202002_spb/cal_20200211/cam_5/s/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_1_0_0
cam_5_0_0
cam_5_0n_0

mtx
ret,mtx,dist = cam_5_0n_1
ret,mtx,dist = cam_5_0n_2
f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')


base_dir = 'e:/Data/Voxels/201911_spb/cal_20191118/'

#from process_board_cloud import *

#cameras  1   2    3    4    5
#angles : 0, 55, 152, 208, 305
cameras = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5']
angles_degrees = [0, 305, 208, 152, 55]

cam = 4
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))



copy_pictures(data_dir)

data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

rot,t = _proc_cal_idx([0,1,2], rotate = lr, VLP32_multi=1., scale=1)
lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,0,angles_degrees[cam]]))
rot,t = _proc_cal_idx([0,1,2], rotate = lidar_rotation, VLP32_multi=1., scale=1)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1., use9 = False):
    
    pts_idx = np.arange(81);                       
    if (use9):
        pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.111, 9, 0.140),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(9,9));
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    
    ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
    
    bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
    
    scatt2d(ax1,imgpts[:,0,:],False,'w','x',size=25)
    scatt2d(ax1,bpts[:,0,:],False,'b','.',size=1)
    
    sum = cv2.norm(imgpts[:,0,:], boards);
    print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
    print ('reproj error =', sum/len(boards))
    return rot,t
rot,t = _proc_cal_idx([0,1,2], rotate = lidar_rotation, VLP32_multi=1., scale=1)
rot,t = _proc_cal_idx([0,1,2], rotate = lidar_rotation, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([0,1,2], rotate = lidar_rotation, VLP32_multi=1.)
jpglist
f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')

rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=1.)
lidar_rotation
jpglist
_board = load_draw_2d_board(jpglist[0],mtx, dist, false,ax1,(9,9));
_board = load_draw_2d_board(jpglist[0],mtx, dist, False,ax1,(9,9));
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',1)
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    if (rotate is not None):
        sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([3], rotate = lidar_rotation, VLP32_multi=2.)
_board = load_draw_2d_board(jpglist[0],mtx, dist, False,ax1,(9,9));
def calc_image_grid(num, base_dir,back,ax=None):
    fname = base_dir + '/{:06}.png'.format(num)
    print(fname);
    return load_draw_2d_board(fname,back,ax)



def load_draw_2d_board(name,  mtx, dist, back,ax=None, shape = None,
                       scale = 1.):
    img = cv2.imread(name,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return draw_2d_board(img, mtx, dist, back, ax, shape, scale)


def draw_2d_board(img, mtx, dist, back=False, ax=None, shape = None, scale = 1.):
    #if (not ax is None):
    #    ax.cla()
    
    #uimg=cv2.undistort(img,mtx,dist)
    if ( scale > 0 and scale != 1):
        img = cv2.resize( img, (0,0), fx = scale, fy = scale)
    
    
    if (not shape is None):
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        
        flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
        ret,corn_xy = cv2.findChessboardCorners(grey,shape, flagCorners)
        if (ret):
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            #cv2.cornerSubPix(grey,corn_xy,shape, (-1,-1),term)
        else:
            return
        cxy=np.array(corn_xy[:,0,:]).astype(np.float64)
    
    else:
        shape = (11,11) #LA recordings with manual corners
        corners = ( (img[:,:,0]<=0) & (img[:,:,1]<=0) & (img[:,:,2]> 254) )
        corn_xy=np.nonzero(corners.transpose())
        cxy=np.array(corn_xy).transpose().astype(np.float64)
    
    # for resized (enlarged) imnages!!!
    if (False and img.shape[0] > 1100):
        cxy = cxy / 2;
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    if (not ax is None):
        ax.imshow(img)
    
    
    _cxy = cxy.copy()    
    
    cxy_u = _cxy; #cv2.undistortPoints(np.array([_cxy]),mtx,dist, R=mtx)[0]
    
    
    #if (not ax is None):
    #    scatt2d(ax,cxy_u, False, None, 'o',50)
    
    top_idx = np.argmin(cxy_u[:,1])
    right_idx = np.argmax(cxy_u[:,0])
    
    top = cxy_u[top_idx]
    cxy_ut = cxy_u-top;
    
    right = cxy_ut[right_idx]
    angle = np.degrees(np.arcsin(right[1]/np.linalg.norm(right)))
    rm = np.matrix(cv2.getRotationMatrix2D((0,0), -angle,1)[:,:2])
    
    
    cxy_rot = np.array(cxy_ut * rm);
    
    order = np.argsort(cxy_rot[:,1])
    
    final_order = np.array([]);
    for i in np.arange(0,shape[0]*shape[1],shape[0]):
        row = cxy_rot[order[i:i+shape[1]]]
        row_order = np.argsort(row[:,0])
        final_order = np.append(final_order,order[row_order+i])
    
    
    if (back):    
        final_order = final_order.reshape(shape).transpose().reshape(-1)    
    cxy_ret = cxy[final_order.astype(np.int32)]
    
    #cxy_ret = cxy_ret[pts_idx_121]
    #_cxy = _cxy[pts_idx_121]
    
    if (not ax is None):
        scatt2d(ax, _cxy, False, None, 'd',30)
        
        for i in range(len(cxy_ret)):
            ax.annotate(str(i+1),np.array(cxy_ret[i]))
    
    return cxy_ret
_board = load_draw_2d_board(jpglist[0],mtx, dist, False,ax1,(9,9));
_board
rot,t = _proc_cal_idx([3], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([0,1,2,3], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([0,1], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([0,2], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([2], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([3], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([5,6], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([6], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([7], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([5,7], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([8], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([9], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([10], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([1,3], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([1,3,8,9], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([9], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([18], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([17], rotate = lidar_rotation, VLP32_multi=2.)
jpglist
rot,t = _proc_cal_idx([19], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([20], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([21], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([15], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([14], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([4], rotate = lidar_rotation, VLP32_multi=2.)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, "camera_name":camera}


cam_calib_dict
save_calib_json("cam_5_n.json", cam_calib_dict)
angle
np.degrees(angle)
rot,t = _proc_cal_idx([4], rotate = lr, VLP32_multi=2.)
lr
rot,t = _proc_cal_idx([3,4], rotate = lr, VLP32_multi=2.)
cam
angles_degrees[cam]
np.eye
np.eye(3)
rot,t = _proc_cal_idx([3,4], rotate = e, VLP32_multi=2.)
e
rot,t = _proc_cal_idx([3], rotate = e, VLP32_multi=2.)
rot,t = _proc_cal_idx([3], rotate = lr, VLP32_multi=2.)
rot,t = _proc_cal_idx([3], rotate = None, VLP32_multi=2.)
rot,t = _proc_cal_idx([3,4], rotate = None, VLP32_multi=2.)
rot,t = _proc_cal_idx([3], rotate = None, VLP32_multi=2.)
rot,t = _proc_cal_idx([4], rotate = None, VLP32_multi=2.)
rot,t = _proc_cal_idx([5], rotate = None, VLP32_multi=2.)
rot,t = _proc_cal_idx([6], rotate = None, VLP32_multi=2.)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=2.)
rot,t = _proc_cal_idx([6], rotate = lidar_rotation, VLP32_multi=2.)
lidar_rotation
rot,t = _proc_cal_idx([4], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([5], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([10], rotate = lidar_rotation, VLP32_multi=2.)
def calc_image_grid(num, base_dir,back,ax=None):
    fname = base_dir + '/{:06}.png'.format(num)
    print(fname);
    return load_draw_2d_board(fname,back,ax)



def load_draw_2d_board(name,  mtx, dist, back,ax=None, shape = None,
                       scale = 1.):
    img = cv2.imread(name,-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return draw_2d_board(img, mtx, dist, back, ax, shape, scale)


def draw_2d_board(img, mtx, dist, back=False, ax=None, shape = None, scale = 1.):
    #if (not ax is None):
    #    ax.cla()
    
    #uimg=cv2.undistort(img,mtx,dist)
    if ( scale > 0 and scale != 1):
        img = cv2.resize( img, (0,0), fx = scale, fy = scale)
    
    
    if (not shape is None):
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        
        flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
        ret,corn_xy = cv2.findChessboardCorners(grey,shape, flagCorners)
        if (ret):
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            #cv2.cornerSubPix(grey,corn_xy,shape, (-1,-1),term)
        else:
            return
        cxy=np.array(corn_xy[:,0,:]).astype(np.float64)
    
    else:
        shape = (11,11) #LA recordings with manual corners
        corners = ( (img[:,:,0]<=0) & (img[:,:,1]<=0) & (img[:,:,2]> 254) )
        corn_xy=np.nonzero(corners.transpose())
        cxy=np.array(corn_xy).transpose().astype(np.float64)
    
    # for resized (enlarged) imnages!!!
    if (False and img.shape[0] > 1100):
        cxy = cxy / 2;
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    if (not ax is None):
        ax.imshow(img)
    
    
    _cxy = cxy.copy()    
    
    cxy_u = cv2.undistortPoints(np.array([_cxy]),mtx,dist, R=mtx)[0]
    
    
    #if (not ax is None):
    #    scatt2d(ax,cxy_u, False, None, 'o',50)
    
    top_idx = np.argmin(cxy_u[:,1])
    right_idx = np.argmax(cxy_u[:,0])
    
    top = cxy_u[top_idx]
    cxy_ut = cxy_u-top;
    
    right = cxy_ut[right_idx]
    angle = np.degrees(np.arcsin(right[1]/np.linalg.norm(right)))
    rm = np.matrix(cv2.getRotationMatrix2D((0,0), -angle,1)[:,:2])
    
    
    cxy_rot = np.array(cxy_ut * rm);
    
    order = np.argsort(cxy_rot[:,1])
    
    final_order = np.array([]);
    for i in np.arange(0,shape[0]*shape[1],shape[0]):
        row = cxy_rot[order[i:i+shape[1]]]
        row_order = np.argsort(row[:,0])
        final_order = np.append(final_order,order[row_order+i])
    
    
    if (back):    
        final_order = final_order.reshape(shape).transpose().reshape(-1)    
    cxy_ret = cxy[final_order.astype(np.int32)]
    
    #cxy_ret = cxy_ret[pts_idx_121]
    #_cxy = _cxy[pts_idx_121]
    
    if (not ax is None):
        scatt2d(ax, _cxy, False, None, 'd',30)
        
        for i in range(len(cxy_ret)):
            ax.annotate(str(i+1),np.array(cxy_ret[i]))
    
    return cxy_ret
rot,t = _proc_cal_idx([10], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([4], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([3], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([1,4], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([4], rotate = lidar_rotation, VLP32_multi=2.)
rot,t = _proc_cal_idx([4], rotate = lidar_rotation, VLP32_multi=2., use9 = True)
rot,t = _proc_cal_idx([1,4], rotate = lidar_rotation, VLP32_multi=2., use9 = True)
rot,t = _proc_cal_idx([1,4], rotate = lr, VLP32_multi=2., use9 = True)
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',1)
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    if (rotate is not None):
        sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return _cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([1,4], rotate = lr, VLP32_multi=2., use9 = True)
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',1)
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    #if (rotate is not None):
    #    sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([1,4], rotate = lr, VLP32_multi=2., use9 = True)
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    if (ax):
        scatt3d(ax,cloud,not overlay,'#1f1f1f','o',1)
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    #if (rotate is not None):
    #    sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return _cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([1,4], rotate = lr, VLP32_multi=2., use9 = True)
rot,t = _proc_cal_idx([1,4], rotate = lr, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([1,4], rotate = None, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = None, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    
    if (ax):
        scatt3d(ax,_cloud,not overlay,'#1f1f1f','o',1)
    
    
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    #if (rotate is not None):
    #    sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return _cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = None, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = None, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = lr, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = None, VLP32_multi=2., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,0]))
rot,t = _proc_cal_idx([0], rotate = lr, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0,4], rotate = lr, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([4], rotate = lr, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([4], rotate = None, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([4], rotate = lr, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([4], rotate = None, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([4], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
copy_pictures('e:/Data/Voxels/202002_spb/cal_20200210/lidar_cam_5/ ')
copy_pictures('e:/Data/Voxels/202002_spb/cal_20200210/lidar_cam_5')
base_dir = 'e:/Data/Voxels/e:/Data/Voxels/202002_spb/cal_20200210/'
rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
csvlist
data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

csvnanes
csvnames
data_dir
csvlist
base_dir = 'e:/Data/Voxels/202002_spb/cal_20200210/'
data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

csvnames
jpgnames
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=2., use9 = False)
cl = 1
rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=1., use9 = False)
VLP32_multi=1
_cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                 _grid=(0.111, 9, 0.140),
                                 delimiter = ',',rotate=rotate,
                                 VLP32_multi = VLP32_multi); 

_board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(9,9));

rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=1., use9 = False)
rotate
lr
lidar_rotation
lr
rot,t = _proc_cal_idx([0], rotate = rotate, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1], rotate = rotate, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,1], rotate = rotate, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([2], rotate = rotate, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([3], rotate = rotate, VLP32_multi=1., use9 = False)
lidar_rotation = eulerAnglesToRotationMatrix(np.radians([0,180,angles_degrees[cam]]))
rot,t = _proc_cal_idx([3], rotate = lidar_rotation, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([3], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,3], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([2], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([3], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,3], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([4], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([5], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,3,5], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([6], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([7], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([8], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([9], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([10], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([3,10], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([9,10], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,3,5,6,9,10], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([12], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,11,12,13], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,11,12,13,5], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([14], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([16], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([17], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([18], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([19], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([20], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([21], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([19,20], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,14], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,16], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,17], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,18], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,19], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,20], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,20], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19,20], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19,20], rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19], rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19,1], rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19], rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19б20], rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19,20], rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([1,3,5,6,9,10,11,12,13,14,15,16,17,18,19,20], rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([1,3,5,6,9,10,11,12,13,14,15,16,17,18,19,20], rotate = lr, VLP32_multi=1., use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_json(base_dir+"Flir_"+camera+"_.json", cam_calib_dict)

save_calib_list_json("cam_5_n.json", [cam_calib_dict])
save_calib_list_json(base_dir+"Flir_"+camera+"_.json", [cam_calib_dict])
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    
    if (ax):
        scatt3d(ax,_cloud,not overlay,'#1f1f1f','o',1)
    
    
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    if (rotate is not None):
        sorted_grid = sorted_grid * rotate.transpose()
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    #rotated_cloud = cloud * box_rot.transpose
    
    return cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([1,3,5,6,9,10,11,12,13,14,15,16,17,18,19,20], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,3], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,3,5,6,9,10,11,12,13,14,15,16,17,18,19,20], rotate = lr, VLP32_multi=1., use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_.json", [cam_calib_dict])

rot,t = _proc_cal_idx([12,13,14,15,16,17,18,19,20], rotate = lr, VLP32_multi=1., use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_.json", [cam_calib_dict])

def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    
    if (ax):
        scatt3d(ax,_cloud,not overlay,'#1f1f1f','o',1)
    
    
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    if (rotate is not None):
        sorted_grid = sorted_grid * rotate.transpose()
    
    return cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([12,13,14], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([12,14,13], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([12,13], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,12,15,16], rotate = lr, VLP32_multi=1., use9 = False)
lr
rotationMatrixToEulerAngles(lr)
lidar_rotation = eulerAnglesToRotationMatrix(np.radians([0,180,angles_degrees[cam]]))
lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,0,angles_degrees[cam]]))
rot,t = _proc_cal_idx([13,12,15,16], rotate = lidar_rotation, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
lr55 = eulerAnglesToRotationMatrix(np.radians([180,0,55]))
rot,t = _proc_cal_idx([13,12], rotate = lr55, VLP32_multi=1., use9 = False)
lr305 = eulerAnglesToRotationMatrix(np.radians([180,0,305]))
rot,t = _proc_cal_idx([13,12], rotate = lr305, VLP32_multi=1., use9 = False)
lr55 = eulerAnglesToRotationMatrix(np.radians([180,0,0])) * eulerAnglesToRotationMatrix(np.radians([0,0,55]))
lr55

lidar_rotation
rot,t = _proc_cal_idx([13,12], rotate = lr55, VLP32_multi=1., use9 = False)
lr55 =  eulerAnglesToRotationMatrix(np.radians([0,0,55])) * eulerAnglesToRotationMatrix(np.radians([180,0,0]))

lr55
lidar_rotation
rot,t = _proc_cal_idx([13,12], rotate = lr55, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
yaml
import yaml
import pyyaml
import yaml
runfile('D:/WORK/Voxels/PyVoxels/process_board_cloud.py', wdir='D:/WORK/Voxels/PyVoxels')
f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')


base_dir = 'e:/Data/Voxels/202002_spb/cal_20200210/'

#from process_board_cloud import *

#cameras  1   2    3    4    5
#angles : 0, 55, 152, 208, 305
cameras = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5']
angles_degrees = [0, 305, 208, 152, 55]

cam = 4
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))



copy_pictures(data_dir)



mtx = cam_cal[1]
dist = cam_cal[2]

data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

import pickle
matplotlib qt
f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')



base_dir = 'e:/Data/Voxels/202002_spb/cal_20200210/'

#from process_board_cloud import *

#cameras  1   2    3    4    5
#angles : 0, 55, 152, 208, 305
cameras = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5']
angles_degrees = [0, 305, 208, 152, 55]

cam = 4
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))



copy_pictures(data_dir)



mtx = cam_cal[1]
dist = cam_cal[2]

data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)


p = pickle.load(open("E:\\Data\\Voxels\\202002_spb\\cal_20200210\\Flir_cam_5_dict_81.p",'b'))
p = pickle.load(open("E:\\Data\\Voxels\\202002_spb\\cal_20200210\\Flir_cam_5_dict_81.p",'rb'))
p
mtx = p['mtx']
dist = p['dist']
pickle.dump([0,mtx,dist],open("cam_5.p","wb"))
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))
cam_cal
cam_cal[1]
mtx = cam_cal[1]
dist = cam_cal[2]

rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,0]))
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1., use9 = False):
    
    pts_idx = np.arange(81);                       
    if (use9):
        pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.111, 9, 0.140),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(9,9));
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    
    ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
    
    bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
    
    scatt2d(ax1,imgpts[:,0,:],False,'w','x',size=25)
    scatt2d(ax1,bpts[:,0,:],False,'b','.',size=1)
    
    sum = cv2.norm(imgpts[:,0,:], boards);
    print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
    print ('reproj error =', sum/len(boards))
    return rot,t
rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

import glob
import os
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitali
import numpy as np
import cv2


from calibrate_camera import *
from process_board_cloud import *
from calib2json import *

from scatt3d import *

rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,-55]))
rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,5]))
lr = eulerAnglesToRotationMatrix(np.radians([180,0,1]))
rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
fe = plt.figure()
cam_5_0n_0 = calibrate('E://Data/Voxels/202002_spb/cal_20200211/cam_5/s/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
dist
mtx
rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,13,12,14,15], rotate = lr, VLP32_multi=1., use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_.json", [cam_calib_dict])


dist
cam5_
cam_5_0n_0
mtx = cam_5_0n_0[1]
dist = cam_5_0n_0[2]
rot,t = _proc_cal_idx([11,13,12,14,15], rotate = lr, VLP32_multi=1., use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_.json", [cam_calib_dict])


lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,0,angles_degrees[cam]]))
rot,t = _proc_cal_idx([11,13], rotate = lidar_rotation, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,5]))
rot,t = _proc_cal_idx([11,13,12,14,15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,12], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,11], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,15]))
rot,t = _proc_cal_idx([13,11], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,5]))
rot,t = _proc_cal_idx([13,11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,12,11], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,-5]))
rot,t = _proc_cal_idx([13,12,11], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,-15]))
rot,t = _proc_cal_idx([13,12,11], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,-25]))
rot,t = _proc_cal_idx([13,12,11], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,-55]))
rot,t = _proc_cal_idx([13,12,11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,12,11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([12,11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13б11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13, 11], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,-15]))
rot,t = _proc_cal_idx([13, 11], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,-1]))
rot,t = _proc_cal_idx([13, 11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13, 12,15,11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13, 12,15,14,11], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-1])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-15])), VLP32_multi=1., use9 = True)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1., use9 = False):
    
    pts_idx = np.arange(81);                       
    if (use9):
        pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.111, 9, 0.140),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(9,9));
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
                print(_grid)
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    
    ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
    
    bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
    
    scatt2d(ax1,imgpts[:,0,:],False,'w','x',size=25)
    scatt2d(ax1,bpts[:,0,:],False,'b','.',size=1)
    
    sum = cv2.norm(imgpts[:,0,:], boards);
    print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
    print ('reproj error =', sum/len(boards))
    return rot,t
rot,t = _proc_cal_idx([13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-15])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-10])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([1,13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([2,13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([1,2,3,13,11], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([1,2,3], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([1,3,5,6], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([1,3,5,6], rotate = eulerAnglesToRotationMatrix(np.radians([180,9,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([1,3,5,6], rotate = eulerAnglesToRotationMatrix(np.radians([180,-9,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19], rotate = eulerAnglesToRotationMatrix(np.radians([180,-9,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([11,13,12,14,15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,12,13,14,15,16,17,18,19], rotate = eulerAnglesToRotationMatrix(np.radians([180,-9,-55])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,13,12,14,15,19], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,-9,-55]))
rot,t = _proc_cal_idx([11,13,12,14,15,19], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,13,12,14,15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,13,12], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([14,15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([14], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([15], rotate = lr, VLP32_multi=1., use9 = False)
def _proc_cal_idx(idx, rotate = None, back=False, VLP32_multi = 1., use9 = False):
    
    pts_idx = np.arange(81);                       
    if (use9):
        pts_idx = np.array([1,5,9,37,41,45,73,77,81])-1
    #idx = cam0_idx
    
    global _cloud
    ax1.cla()
    ax3.cla()
    
    clouds = np.empty((0,3),np.float64)
    boards = np.empty((0,2),np.float64)
    
    for cl in idx:
         try:
            _cloud,_grid, flat, box = calc_cloud_grid_f(csvlist[cl],ax3, True, 
                                             _grid=(0.111, 9, 0.140),
                                             delimiter = ',',rotate=rotate,
                                             VLP32_multi = VLP32_multi); 
            _board = load_draw_2d_board(jpglist[cl],mtx, dist, back,ax1,(9,9));
            
            #_cloud = _cloud * rm90
            
            _grid = _grid[pts_idx]
            _board = _board[pts_idx]                                        
            if (_grid is not None and _board is not None):                                        
                clouds=np.append(clouds, _grid ,0);
                boards=np.append(boards, _board,0)
                print (csvnames[cl], len(_grid), len(_board))
                #print(_grid)
            else:
                throw(0)
         except:
            print("error: ",csvnames[cl])
    
    
    ret, rot, t = cv2.solvePnP(clouds,boards,mtx,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    imgpts, jac = cv2.projectPoints(clouds, rot, t, mtx, dist)
    
    bpts, _ = cv2.projectPoints(_cloud, rot, t, mtx, dist)
    
    scatt2d(ax1,imgpts[:,0,:],False,'w','x',size=25)
    scatt2d(ax1,bpts[:,0,:],False,'b','.',size=1)
    
    sum = cv2.norm(imgpts[:,0,:], boards);
    print(idx,'\r\n', rot.transpose(),'\r\n', t.transpose())
    print ('reproj error =', sum/len(boards))
    return rot,t
rot,t = _proc_cal_idx([14,15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([14], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([15], rotate = lr, VLP32_multi=1., use9 = False)
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    
    #if (ax):
    #    scatt3d(ax,_cloud,not overlay,'#1f1f1f','.',1)
    
    
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    if (rotate is not None):
        sorted_grid = sorted_grid * rotate.transpose()
    
    return cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx([14], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([15,14], rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,0,-55]))
rot,t = _proc_cal_idx([15,14], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([14,15], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-5])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([14,15], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([14,15], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([14,15], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,15])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,13,12,14,15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,2,3,11,13,12,14,15], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,2,3,11,13,15,14,12,0], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,2,3,11,13,15,14,12,0,4], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,2,3,11,13,15,14,12,0,4,5,6,7,8,9,10], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,13,12,14,15,16,17,18,19], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([11,13,14,15,16,17,18,19,20,12], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,1,2], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,15])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,1], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,15])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,3], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,15])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1,3], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,15])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([2,3], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,15])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([3], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,15])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([2], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,15])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,1,2], rotate = eulerAnglesToRotationMatrix(np.radians([180,0,-55])), VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,1,2], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([1], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,1], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,2], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,2,3], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,2,3,4], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,2,3,5], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,2,3,5,6], rotate = lr, VLP32_multi=1., use9 = False)
last = np.arange(11,21)
last
last = [11:20]
last = [i for i in range(11,21)]
last
rot,t = _proc_cal_idx([0,2,3,5,6]+last, rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,2,3,5,6]+last, rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([0,2,3,5,6], rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx(last, rotate = lr, VLP32_multi=1., use9 = False)
def calc_cloud_grid(num, base_dir, ax=None, overlay=False,London=False, _grid = (0.0597, 11, 0.211)):
    fname = base_dir + '\\{:06}.csv'.format(num)
    return calc_cloud_grid_f(fname, ax, overlay, London, _grid)


pts_idx_121=np.array([1,6,11,56,61,66,111,116,121])-1



def calc_cloud_grid_f(fname, ax=None, overlay=False,London=False, 
                      _grid = (0.0597, 11, 0.211),delimiter = ',', 
                      rotate=None, VLP32_multi=1.):
    cloud = read_cloud_file(fname, delimiter)[0]
    
    #NB!! Only for recordings 20191118 (VLP32 with wrong resolution)
    cloud = cloud * VLP32_multi
    
    
    _cloud = cloud.copy()
    if (rotate is not None):
        _cloud = _cloud * rotate
    
    if (ax):
        scatt3d(ax,_cloud,not overlay,'#2f2f2f','.',1)
    
    
    _avg, _rot = get_cloud_rotation(_cloud)
    
    flat = rotate_cloud(_cloud,_avg,_rot)
    
    box = get_box(flat)
    
    p0,box_rot = get_box_rotation(box)
    
    #if (London):
    #    grid =(0.04, 11, 0.14)# - LONDON
    
    grid = build_grid(_grid[0], _grid[1], _grid[2])
    
    #first rotation - to flat box
    rotated_grid = grid*box_rot.transpose()+p0
    
    #second rotation - to the original box image
    rotated_grid = np.array(rotated_grid * _rot + _avg)
    #rotated_grid = np.array(rotated_grid*la.pinv(rot).transpose() + avg)
    
    sorted_grid = sort_grid(rotated_grid,_grid[1])
    
    #sorted_grid = sorted_grid[pts_idx_121]
    
    
    if (ax):
        scatt3d(ax,[0,0,0],False)
        scatt3d(ax,sorted_grid,False,None,'d',49)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #left_border = np.ceil( max(cloud[:,1]) * 5.) / 5.
        #right_border = np.floor( min(cloud[:,1]) * 5.) / 5.
        #ax.set_ylim(right_border, left_border)
        #ax_3d.set_zlim(-1,1)
        #ax.axis('equal')
        for i in range(len(sorted_grid)):
            ax.text(sorted_grid[i,0],sorted_grid[i,1],sorted_grid[i,2], str(i+1)  )
    
    
    if (rotate is not None):
        sorted_grid = sorted_grid * rotate.transpose()
    
    return cloud, sorted_grid, flat, box
rot,t = _proc_cal_idx(last, rotate = lr, VLP32_multi=1., use9 = False)
lr = eulerAnglesToRotationMatrix(np.radians([180,-9,-55]))
rot,t = _proc_cal_idx(last, rotate = lr, VLP32_multi=1., use9 = False)
rot,t = _proc_cal_idx([0,2,3,5,6]+last, rotate = lr, VLP32_multi=1., use9 = True)
rot,t = _proc_cal_idx([0,2,3,5,6]+last, rotate = lr, VLP32_multi=1., use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

save_calib_list_json(base_dir+"Flir_"+camera+"_all.json", [cam_calib_dict])
rot,t = _proc_cal_idx(last, rotate = lr, VLP32_multi=1., use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

save_calib_list_json(base_dir+"Flir_"+camera+"_last.json", [cam_calib_dict])
cam_4_0_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_4/sel', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_4_0_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_4/sel', model=0, draw_fig = fe, scale = 1.0, nSamples=20, sqSide=39.6)
cam_5_0n_0
cam_5_0n_1
cam_5_0n_0
cam_4_0_1 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_4/sel', model=0, draw_fig = fe, scale = 1.0, nSamples=15, sqSide=39.6)
cam_4_0_2 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_4/sel', model=0, draw_fig = fe, scale = 1.0, nSamples=15, sqSide=39.6)
cam_4_0_3 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_4/sel', model=0, draw_fig = fe, scale = 1.0, nSamples=15, sqSide=39.6)
pickle.dump(cam_4_0_0,open("cam_4.p","wb"))
cam_1_0_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_1_0_1 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data', model=0, draw_fig = fe, scale = 1.0, nSamples=30, sqSide=39.6)
cam_1_0_2 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data', model=0, draw_fig = fe, scale = 1.0, nSamples=30, sqSide=39.6)
cam_1_0_3 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_1/data', model=0, draw_fig = fe, scale = 1.0, nSamples=35, sqSide=39.6)
pickle.dump(cam_1_0_0,open("cam_1.p","wb"))
cam_2_0_0 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_2/data', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_2_0_1 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_2/data', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_2_0_2 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_2/data', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_2_0_3 = calibrate('E://Data/Voxels/201911_spb/cal_20191118/cam_2/data', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=39.6)
cam_2_0_2
cam_2_0_3
cam_2_0_1
pickle.dump(cam_2_0_1,open("cam_2.p","wb"))
copy_pictures('e:/Data/Voxels/202002_spb/cal_20200210/lidar_cam_1')
runfile('D:/WORK/Voxels/PyVoxels/utils/copy_pictures.py', wdir='D:/WORK/Voxels/PyVoxels/utils')
copy_pictures('e:/Data/Voxels/202002_spb/cal_20200210/lidar_cam_1')
copy_pictures('e:/Data/Voxels/202002_spb/cal_20200210/lidar_cam_2')
copy_pictures('e:/Data/Voxels/202002_spb/cal_20200210/lidar_cam_4')
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))

mtx = cam_cal[1]
dist = cam_cal[2]

data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)


lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,-9,-angles_degrees[cam]]))

rot,t = _proc_cal_idx([4,6,10,1], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([0,1], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([2], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([1,2], rotate = lidar_rotation, VLP32_multi=1. )
lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,0,angles_degrees[cam]]))
rot,t = _proc_cal_idx([1,2], rotate = lidar_rotation, VLP32_multi=1. )
cam
cam = 0
rot,t = _proc_cal_idx([1,2], rotate = lidar_rotation, VLP32_multi=1. )
data_dir
camera
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))



#copy_pictures(data_dir)



mtx = cam_cal[1]
dist = cam_cal[2]

data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)


lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,-9,-angles_degrees[cam]]))

data_dir
rot,t = _proc_cal_idx([1,2], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([0,1,2], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([0,2], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([1,2], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([1,2,3], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([1,2,3,4,5,6], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([1,2,3,4], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([1,2,3,5], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([1,2,3,5,6], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([5,6], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([6], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([7], rotate = lidar_rotation, VLP32_multi=1. )
cam
angles_degrees[cam]
lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,-9,angles_degrees[cam]]))
rot,t = _proc_cal_idx([7], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([8], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([7], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([8], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([9], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([10], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([11], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([12], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([12,10], rotate = lidar_rotation, VLP32_multi=1. )
lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,-9,180]))
rot,t = _proc_cal_idx([12,10], rotate = lidar_rotation, VLP32_multi=1. )
lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,0,angles_degrees[cam]]))
rot,t = _proc_cal_idx([12,10], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([10], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([12], rotate = lidar_rotation, VLP32_multi=1. )
rot,t = _proc_cal_idx([12], rotate = lidar_rotation, VLP32_multi=1. , use9 = True)
rot,t = _proc_cal_idx([12,10], rotate = lidar_rotation, VLP32_multi=1. , use9 = True)
rot,t = _proc_cal_idx([12,11,10], rotate = lidar_rotation, VLP32_multi=1. , use9 = True)
rot,t = _proc_cal_idx([13], rotate = lidar_rotation, VLP32_multi=1. , use9 = True)
rot,t = _proc_cal_idx([14], rotate = lidar_rotation, VLP32_multi=1. , use9 = True)
rot,t = _proc_cal_idx([15], rotate = lidar_rotation, VLP32_multi=1. , use9 = True)
rot,t = _proc_cal_idx([15], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,1515], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,15], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,15,10,12], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,15,10], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,15,11], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,15,12], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,15,12], rotate = lidar_rotation, VLP32_multi=1. , use9 = Treu)
rot,t = _proc_cal_idx([13,14,15,12], rotate = lidar_rotation, VLP32_multi=1. , use9 = True)
rot,t = _proc_cal_idx([13,14,15,0], rotate = lidar_rotation, VLP32_multi=1. , use9 = True)
rot,t = _proc_cal_idx([13,14,15,0], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,15,0,16], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
dist
cam_1_0_0
cam_1_0_1
cam_1_0_2
cam_1_0_3
ret,mtx,dist = cam_1_0_2
rot,t = _proc_cal_idx([13,14,15,0,16], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
ret
rot
rot,t = _proc_cal_idx([13,14,15,0], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13,14,15], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([4], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,6], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,7], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,10], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,10,11], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,10,11,12], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,10,11,13], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,10,11,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,10,11,14,13], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,10,11,14,15], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,8,10,11,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

save_calib_list_json(base_dir+"Flir_"+camera+"_14.json", [cam_calib_dict])

rot,t = _proc_cal_idx([0,1,2,3,5,8,10,11,14,15,12], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

save_calib_list_json(base_dir+"Flir_"+camera+"_12.json", [cam_calib_dict])

s = ''
s.empty()
len(s)
cam = 2
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))

camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))

cam = 1
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))

mtx = cam_cal[1]
dist = cam_cal[2]

data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)


lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,-9,-angles_degrees[cam]]))


rot,t = _proc_cal_idx([0], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([1], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([1,0], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([2], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([2,3], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([3], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([4], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([5], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,4,5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+".json", [cam_calib_dict])

dist
cam_2_0_0
cam_2_0_1
cam_2_0_2
cam_2_0_3
save_calib_list_json(base_dir+"Flir_"+camera+"_1.json", [cam_calib_dict])
ret,mtx,dist = cam_2_0_2
rot,t = _proc_cal_idx([0,1,2,3,5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

#pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_2.json", [cam_calib_dict])

ret,mtx,dist = cam_2_0_3
rot,t = _proc_cal_idx([0,1,2,3,5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

#pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_3.json", [cam_calib_dict])


cam = 3
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])
#base_dir = 'e:/data/Voxels/201903_USA/20190321_cal_cam1/'
cam_cal = pickle.load(open(base_dir + camera + '.p','rb'))



#copy_pictures(data_dir)



mtx = cam_cal[1]
dist = cam_cal[2]

data_dir = base_dir + 'lidar_' + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)


lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,-9,-angles_degrees[cam]]))

camera
rot,t = _proc_cal_idx([0,1,2,3,5,6,7,8,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,8], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,10], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,10,8], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,11], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,12], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,15], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,13], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,12,13,14,15], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14,11], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14,12], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([10], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([11], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([12], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([13], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14,8], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([14,9], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
dist
cam_4_0_0
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

#pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_0.json", [cam_calib_dict])


ret,mtx,dist = cam_4_0_1
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

#pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_1.json", [cam_calib_dict])

ret,mtx,dist = cam_4_0_2
cam_4_0_0
cam_4_0_1
cam_4_0_2
cam_4_0_3
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

#pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_2.json", [cam_calib_dict])


ret,mtx,dist = cam_4_0_3
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14,8], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

#pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_3.json", [cam_calib_dict])

camera
cam_4_b_0 = calibrate('E://Data/Voxels/202002_spb/cal_20200210/lidar_cam_4/s/', model=0, draw_fig = fe, scale = 1.0, nSamples=15, sqSide=111.0, nx = 8, ny = 8)
csvlist
jpglist
img = cv2.imread(jpglist[0])
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.imshow(img)
flagCorners = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH 
ret, corners = cv2.findChessboardCorners(grey, (nx, ny), flagCorners)
ret, corners = cv2.findChessboardCorners(grey, (8, 8), flagCorners)
ret
corners
img = cv2.imread(jpglist[10])
plt.imshow(img)
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(grey, (8, 8), flagCorners)
ret, corners = cv2.findChessboardCorners(grey, (9, 9), flagCorners)
ret
corners
cam_4_b_0 = calibrate('E://Data/Voxels/202002_spb/cal_20200210/lidar_cam_4/s/', model=0, draw_fig = fe, scale = 1.0, nSamples=20, sqSide=111.0, nx = 9, ny = 9)
cam_4_0
cam_4_0_0
cam_4_0_1
cam_4_b_1 = calibrate('E://Data/Voxels/202002_spb/cal_20200210/lidar_cam_4/s/', model=0, draw_fig = fe, scale = 1.0, nSamples=15, sqSide=111.0, nx = 9, ny = 9)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
ret,mtx,dist = cam_4_b_0
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')

rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,10,11,12,13,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
ret,mtx,dist = cam_4_b_1
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,9,14], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

#pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_b1.json", [cam_calib_dict])

b141_cam_1_0 = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=111.0, nx = 9, ny = 9)
fe = plt.figure()
b141_cam_1_1 = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/data/', model=0, draw_fig = fe, scale = 1.0, nSamples=40, sqSide=111.0, nx = 9, ny = 9)
b141_cam_1_71 = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/data/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9)
b141_cam_1_71 = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/large/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9)
b141_cam_1_near = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/large/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9)
b141_cam_1_near = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/near/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9)
b141_cam_1_middle = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/middle/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9)
b141_cam_1_not_far = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/not_far/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9)
b141_cam_1_0
b141_cam_1_1
b141_cam_1_near
b141_cam_1_middle
b141_cam_1_not_far
b141_cam_1_not_far_1 = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/not_far/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9, step=2)
b141_cam_1_not_far_2 = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/not_far/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9, step=2)
b141_cam_1_s0 = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/not_far/', model=0, scale = 1.0, sqSide=39.6,step=2)
b141_cam_1_s0 = calibrate('E://Data/Voxels/202002_spb/calib_20200217/cam_1/' , model=0, scale = 1.0, sqSide=39.6,step=2)
shape
b141_cam_1_s0 = calibrate('E://Data/Voxels/202002_spb/calib_20200217/cam_1/' , model=0, scale = 1.0, sqSide=39.6,step=2, nx=9, ny=6)
b141_cam_1_not_far_2 = calibrate('E://Data/Voxels/202002_spb/cal_20200219/cam_1/not_far/', model=0, scale = 1.0, sqSide=111.0, nx = 9, ny = 9, step=2)
b141_cam_1_s0 = calibrate('E://Data/Voxels/202002_spb/calib_20200217/cam_1/s/' , model=0, scale = 1.0, sqSide=39.6,step=2, nx=9, ny=6)
b141_cam_1_s1 = calibrate('E://Data/Voxels/202002_spb/calib_20200217/cam_1/s/' , model=0, scale = 1.0, sqSide=39.6,step=2, nx=9, ny=6)
b141_cam_1_s1
b141_cam_1_s0
b141_cam_1_s1 = calibrate('E://Data/Voxels/202002_spb/calib_20200217/cam_1/s/' , model=0, scale = 1.0, sqSide=39.6, nSamples=40, nx=9, ny=6)
b141_cam_1_s1 = calibrate('E://Data/Voxels/202002_spb/calib_20200217/cam_1/s/' , model=0, scale = 1.0, sqSide=39.6,step=2, nx=9, ny=6)
b141_cam_1_middle
b141_cam_1_near
b141_cam_1_not_far
b141_cam_1_not_far_1
b141_cam_1_not_far_2
b141_cam_1_s1
colors2labels_o('e:/data/segm/202002/20200221 Correct SF/' ,'color_mask','labels',ont[0])
pickle.dump(b141_cam_1_0,open("cam_b141_cam_1_0.p","wb"))
pickle.dump(b141_cam_1_1,open("cam_b141_cam_1_1.p","wb"))
pickle.dump(b141_cam_1_71,open("cam_b141_cam_1_71.p","wb"))
pickle.dump(b141_cam_1_middle,open("cam_b141_cam_1_middle.p","wb"))
pickle.dump(b141_cam_1_near,open("cam_b141_cam_1_near.p","wb"))
pickle.dump(b141_cam_1_not_far,open("cam_b141_cam_1_not_far.p","wb"))
pickle.dump(b141_cam_1_not_far1,open("cam_b141_cam_1_not_far1.p","wb"))
pickle.dump(b141_cam_1_not_far_1,open("cam_b141_cam_1_not_far_1.p","wb"))
pickle.dump(b141_cam_1_not_far_2,open("cam_b141_cam_1_not_far_2.p","wb"))
pickle.dump(b141_cam_1_s0,open("cam_b141_cam_1_s0","wb"))
pickle.dump(b141_cam_1_s1,open("cam_b141_cam_1_s1","wb"))
globals
globals()
g = globals()
g
g.keys()
[i for i in g if 'box141' in i]
[i for i in g if 'box' in i]
[i for i in g if 'cam' in i]
[i for i in g if 'b141' in i]
b141_keys = [i for i in g if 'b141' in i]
g[0]
g[b141_keys[0]]
b141_dict = {k:g[k] for k in b141_keys}
b141_dict
pickle.dump(b141_dict,open("b141_cam_1","wb"))
[i for i in g if 'cam_1' in i]
[i for i in g if '^cam_1' in i]
[i for i in g if '\^cam_1' in i]
[i for i in g if i.startswith('cam_1')]
cam_1_dict = {k:g[k] for k in [i for i in g if i.startswith('cam_1')]}
cam_1_dict
pickle.dump(cam_1_dict,open("cam_1","wb"))
cam_1_dict
cam_2_dict = {k:g[k] for k in [i for i in g if i.startswith('cam_2')]}
pickle.dump(cam_2_dict,open("cam_2","wb"))
cam_4_dict = {k:g[k] for k in [i for i in g if i.startswith('cam_4')]}
pickle.dump(cam_4_dict,open("cam_4","wb"))
cam_5_dict = {k:g[k] for k in [i for i in g if i.startswith('cam_5')]}
box140_cam_1_dict = {k:g[k] for k in [i for i in g if i.startswith('cam_1')]}
pickle.dump(box140_cam_1_dict,open("box140_cam_1","wb"))
box140_cam_2_dict = {k:g[k] for k in [i for i in g if i.startswith('cam_2')]}
pickle.dump(box140_cam_2_dict,open("box140_cam_2","wb"))
box140_cam_4_dict = {k:g[k] for k in [i for i in g if i.startswith('cam_4')]}
pickle.dump(box140_cam_4_dict,open("box140_cam_4","wb"))
box140_cam_5_dict = {k:g[k] for k in [i for i in g if i.startswith('cam_5')]}
pickle.dump(box140_cam_5_dict,open("box140_cam_5","wb"))
pwd
box141_cam_1_
box141_cam_1_0
b141_cam_1_0
b141_cam_1_s0
cam_cal = b141_cam_1_0
base_dir
f1 = plt.figure(1)
f3 = plt.figure(3)

ax1= f1.gca()
ax3=f3.add_subplot(111,projection='3d')


base_dir = 'e:/Data/Voxels/202002_spb/cal_20200219/'

cam
cam=0
cameras = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5']
angles_degrees = [0, 305, 208, 152, 55]

cam = 0
camera = cameras[cam]
angle = np.radians(angles_degrees[cam])

data_dir = base_dir + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)

csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)


data_dir = base_dir + camera + '/'
csvlist = glob.glob(data_dir + 's/*.txt')
jpglist = glob.glob(data_dir + 's/*.jpg')

#names check

csvnames = [os.path.basename(csv).split('.')[0] for csv in csvlist]
jpgnames = [os.path.basename(jpg).split('.')[0] for jpg in jpglist]

assert (csvnames == jpgnames)


lidar_rotation = eulerAnglesToRotationMatrix(np.radians([180,-9,-angles_degrees[cam]]))

rot,t = _proc_cal_idx([0,1,2,3,4,5], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,5,6,7,8,9,10], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,6,7,8,9,10], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
arange(20,34)
np.arange(20,34)
rot,t = _proc_cal_idx(np.arange(20,34), rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([33,34], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx(np.arange(20,25), rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([20,21], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([20,21,22], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([20,21,22,23], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([20,21,22,23,24], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([24], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([25], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([26], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
rot,t = _proc_cal_idx([20,21,22,23,26], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
[0:4,30]
rot,t = _proc_cal_idx(np.arange(5)+[20,21,22,23,26], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
list(np.arange(5))
list(np.arange(5)) + 18
list(np.arange(5)) + [18]
rot,t = _proc_cal_idx(list(np.arange(5)) + list (np.arange(6,24)) +[26], rotate = lidar_rotation, VLP32_multi=1. , use9 = False)
cam_calib_dict = {"mtx":mtx, "dist":dist, "rot":rot, "t":t, "camera_center":angle, 
                   "camera_name":camera}    

#pickle.dump(cam_calib_dict,open(base_dir+"Flir_"+camera+"_dict_81.p","wb"))


save_calib_list_json(base_dir+"Flir_"+camera+"_b141.json", [cam_calib_dict])
