import argparse
import cv2
import numpy as np
import dlib
import scipy
import cvlib as cv
import matplotlib.pyplot as plt

def track_face(img1_gray,tracker,tracker1,mode):
    hog_detector=dlib.get_frontal_face_detector()
    hog_predictor=dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    rects=hog_detector(img1_gray,2)
    no_of_faces=len(rects)
    bbox=[]
    if mode==1:
        ok, bbox1 = tracker.update(img1_gray)
        box1=np.zeros(4)
        box1[0]=bbox1[0]
        box1[1]=bbox1[1]
        box1[2]=bbox1[0]+bbox1[2]
        box1[3]=bbox1[1]+bbox1[3]
        bbox.append(dlib.rectangle(int(box1[0]),int(box1[1]),int(box1[2]),int(box1[3])))
        return no_of_faces,hog_predictor,bbox
    if mode==2:
        box1=np.zeros(4)
        box2=np.zeros(4)
        ok, bbox1 = tracker.update(img1_gray)
        ok, bbox2 = tracker1.update(img1_gray)
        box1[0]=bbox1[0]
        box1[1]=bbox1[1]
        box2[0]=bbox2[0]
        box2[1]=bbox2[1]
        
        box1[2]=bbox1[0]+bbox1[2]
        box1[3]=bbox1[1]+bbox1[3]
        box2[2]=bbox2[0]+bbox2[2]
        box2[3]=bbox2[1]+bbox2[3]
        # print(box2)
        bbox.append(dlib.rectangle(int(box1[0]),int(box1[1]),int(box1[2]),int(box1[3])))
        bbox.append(dlib.rectangle(int(box2[0]),int(box2[1]),int(box2[2]),int(box1[3])))
        # bbox.append(box1)
        # bbox.append(box2)
        return no_of_faces,hog_predictor,bbox

# for face in faces:    
    # (startX,startY) = face[0],face[1]
    # (endX,endY) = face[2],face[3]    # draw rectangle over face
    # cv2.rectangle(img1, (startX,startY), (endX,endY), (0,255,0), 2)
def detect_face(img1_gray,path):
    hog_detector=dlib.get_frontal_face_detector()
    
    # hog_predictor=dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    hog_predictor=dlib.shape_predictor(path)
    rects=hog_detector(img1_gray,2)
    
    no_of_faces=len(rects)
    return no_of_faces,hog_predictor,rects
    
def detect_face_cnn(img1_gray,img1,path,mode):
    hog_detector=dlib.get_frontal_face_detector()
    
    # hog_predictor=dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    hog_predictor=dlib.shape_predictor(path)
    rects=hog_detector(img1_gray,2)
    face, confidences = cv.detect_face(img1,0.29)
    rects=[]
    # print(face)
    if mode==2:
        if len(face)!=2:
            return 0,hog_predictor,rects
        rects.append(dlib.rectangle(int(face[0][0]),int(face[0][1]),int(face[0][2]),int(face[0][3])))
        rects.append(dlib.rectangle(int(face[1][0]),int(face[1][1]),int(face[1][2]),int(face[1][3])))
    if mode==1:
        if len(face)==0:
                   return 0,hog_predictor,rects
        rects.append(dlib.rectangle(int(face[0][0]),int(face[0][1]),int(face[0][2]),int(face[0][3])))
        
    # faces, confidences = cv.detect_face(img1_gray,0.25)
    no_of_faces=len(rects)
    return no_of_faces,hog_predictor,rects


# def landmarks_extraction(img1_gray,hog_predictor,rects):
#     # bounding_box=[]
#     # multi_img_face_lm=[]

#     # landmarks_index_list=[]
#     landmarks_index={}
#     x_=(hog_predictor(img1_gray,rects))
#     facial_landmarks=np.array([[x_.part(i).x ,x_.part(i).y] for i in range(68) ])
#     #if iter_landmarks==1:
#     for ind,val in enumerate(facial_landmarks):
#         landmarks_index[tuple(val)]=ind
#     # landmarks_index_list.append(landmarks_index)
#     # multi_img_face_lm.append(facial_landmarks)
#     bounding=[]

#     bounding.append((rects.left(),rects.top(),rects.right(),rects.bottom()))    

#     # bounding_box.append(x)
#     return bounding,landmarks_index,facial_landmarks
def landmarks_extraction(img1_gray,hog_predictor,rects):
    # bounding_box=[]
    # multi_img_face_lm=[]

    # landmarks_index_list=[]
    landmarks_index={}
    x_=(hog_predictor(img1_gray,rects))
    facial_landmarks=np.array([[x_.part(i).x ,x_.part(i).y] for i in range(68) ])
    #if iter_landmarks==1:
    for ind,val in enumerate(facial_landmarks):
        landmarks_index[tuple(val)]=ind
    # landmarks_index_list.append(landmarks_index)
    # multi_img_face_lm.append(facial_landmarks)
    bounding=[]

    bounding.append((rects.left(),rects.top(),rects.right(),rects.bottom()))     

    # bounding_box.append(x)
    return bounding,landmarks_index,facial_landmarks

def get_bounding_box(facial_landmarks):
    x,y,w,h=(cv2.boundingRect(cv2.convexHull(facial_landmarks)))
    
    return ((x,y),(x+w,y+h)),cv2.Subdiv2D((x,y,x+w,y+h))
def disp_landmarks_face(img1,facial_landmarks,rect):
    for i in range(68): 
        cv2.circle(img1,facial_landmarks[i][:],color=[255,255,255],radius=5,thickness=2)
    img1=cv2.rectangle(img1,rect[0], rect[1],color=[255,255,255],thickness=5)
    return img1

def delunay_triangle(subdiv2,multi_img_face_lm):
    delunay_traingle_pts_list=[]
    # print((multi_img_face_lm[0].shape))
    iter_for_sub=0
    # subdiv2[0].insert((int(multi_img_face_lm[0][0][0]),int(multi_img_face_lm[0][0][1])))
    for sub in subdiv2:

        #print(len(multi_img_face_lm))
        i=1
        for p in multi_img_face_lm[iter_for_sub].tolist():

            # print(i)
            #print(p)
            i=i+1
            sub.insert((p[0],p[1]))
        delunay_traingle_pts_list.append(sub.getTriangleList())
        iter_for_sub=1
    return delunay_traingle_pts_list[0]

def check_point(pts,box):
    (xmin,ymin),(xmax,ymax)=box

    if pts[0]>=xmin and pts[1]>=ymin and pts[0]<=xmax and pts[1]<=ymax:
        return True
    else:
        return False
def matching_delunay_triangles(delunay_triangle_pts_list,rect_,landmarks_index_list,img1,multi_img_face_lm):

    sorted_delunay_lists=[]
    sorting_index_for_delunay_list=[]
    iter_=0
    #for delunay_traingle_points,landmark_index_ in zip(delunay_traingle_pts_list,landmarks_index_list):
    index_array_img=[]
    sorted_delunay_pts1=[]
    sorted_delunay_pts2=[]
    sorting_index_for_delunay=[]
    for p in delunay_triangle_pts_list:
        pts1=(int(p[0]),int(p[1]))
        pts2=(int(p[2]),int(p[3]))    
        pts3=(int(p[4]),int(p[5]))        
        #print(pts1)
        box=rect_[iter_]
        if check_point(pts1,box) and check_point(pts2,box) and check_point(pts3,box)  :
            pts_vector1=np.array([pts1,pts2,pts3])
            pts_vector2=np.zeros(pts_vector1.shape,dtype=np.int32)
            #print(pts_vector1)
            #index_array=np.zeros(3)
            #for 
            index_array=[]
            index=landmarks_index_list[0].get(tuple(pts1),False)
            if index:
                #index_array[0]=index
                index_array.append(index)
                pts_vector2[0]=multi_img_face_lm[1][index]

            else:
                continue
            index=landmarks_index_list[0].get(tuple(pts2),False)
            #pts_vector2[0]=multi_img_face_lm[0][index]
            if index:
                # index_array[1]=index
                index_array.append(index)
                pts_vector2[1]=multi_img_face_lm[1][index]
            else:
                continue
            index=landmarks_index_list[0].get(tuple(pts3),False)
            if index:
                # index_array[2]=index
                index_array.append(index)
                pts_vector2[2]=multi_img_face_lm[1][index]
                #print(pts_vector)
            else:
                continue
            #index_add.append()
            if len(index_array)!=3:
                print('False')
            index_array=np.array(index_array)
            index_to_sort=np.argsort(index_array)
            index_array=index_array[index_to_sort]
            pts_vector1=pts_vector1[index_to_sort,:]
            pts_vector2=pts_vector2[index_to_sort,:]
            #print(pts_vector.shape)
            inds=''
            #print(index_array)

            for ind_x in index_array:
                ind_x=int(ind_x)
                inds=inds+str(ind_x)
            index_array_img.append(int(inds))
            #index_array_img.append(index_array)
            sorted_delunay_pts1.append(pts_vector1)
            sorted_delunay_pts2.append(pts_vector2)
            # cv2.line(img1,pts1,pts2,[255,0, 255],1)
            # cv2.line(img1,pts3,pts2,[255,0,255],1)
            # cv2.line(img1,pts3,pts1,[255,0,255],1)
            # cv2.line(img1,(pts_vector2[0,0],pts_vector2[0,1]),(pts_vector2[1,0],pts_vector2[1,1]),[255,0, 255],1)
            # cv2.line(img1,(pts_vector2[2,0],pts_vector2[2,1]),(pts_vector2[1,0],pts_vector2[1,1]),[255,0,255],1)
            # cv2.line(img1,(pts_vector2[2,0],pts_vector2[2,1]),(pts_vector2[0,0],pts_vector2[0,1]),[255,0,255],1)
    iter_=iter_+1
    sorted_delunay_pts1=np.array(sorted_delunay_pts1)
    sorted_delunay_pts2=np.array(sorted_delunay_pts2)
    sorted_delunay_lists.append(sorted_delunay_pts1)
    sorted_delunay_lists.append(sorted_delunay_pts2)

    index_array_img=np.array(index_array_img)

    sorting_index_for_delunay_list.append(index_array_img)
    return img1,sorted_delunay_lists,sorting_index_for_delunay_list
    # def delunay_triangles()
def global_cord_img(x_,y_,h_,w_):
    l1=np.arange(x_,x_+h_)
    l2=np.arange(y_,y_+w_)
    a=np.meshgrid(l1,l2)
    a=np.array(a)
    #a
    a=np.moveaxis(a,0,-1)
    ones=np.ones((a.shape[0],a.shape[1],1))
    g_cord=np.concatenate([a,ones],-1)
    return g_cord
#delunay_traingle_points[0]
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);
    #coords=np.array([[x0,x1,x1,x0],[y0,y1,y0,y1]])
    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    wa=np.tile(wa,(3,1)).T
    wb=np.tile(wb,(3,1)).T
    wc=np.tile(wc,(3,1)).T
    wd=np.tile(wd,(3,1)).T
    return wa*Ia + wb*Ib + wc*Ic + wd*Id
# bounded_array_f=0
def image_warping(del_tri_pts1,del_tri_pts2,img2,img3,img2_black,img3_black):

    x_,y_,h_,w_=cv2.boundingRect((del_tri_pts1.reshape(3,2)))

    bounded_array=global_cord_img(x_-2,y_-2,h_+2,w_+2)
    m,n,z=bounded_array.shape
    bounded_array=np.reshape(bounded_array,(m*n,z))

    B=np.ones((3,3))
    B[0:2,:]=del_tri_pts1.reshape(3,2).T
    if np.linalg.det(B)!=0:
        B_inv=np.linalg.inv(B)
    else:
        B_inv=np.linalg.pinv(B)

    bayesian_coordinates=(B_inv@bounded_array.T).T

    zero_arr_1=np.all(bayesian_coordinates<=1.01 ,axis=1)
    zero_arr_2=np.all(bayesian_coordinates>=-0.01,axis=1)
    # print(zero_arr_1.shape)
    zero_arr=(zero_arr_1 * zero_arr_2)

    zero_condition_1=np.stack((zero_arr,zero_arr,zero_arr)).T

    mask=zero_condition_1
    x11=bayesian_coordinates
    bayesian_coordinates_f=np.multiply(bayesian_coordinates,mask)
    bounded_array_f=np.multiply(bounded_array,mask)
    bounded_array_f=bounded_array_f[~np.all((bayesian_coordinates_f)==0,axis=1)]
    bayesian_coordinates_f=bayesian_coordinates_f[~np.all((bayesian_coordinates_f)==0,axis=1)]

    A=np.ones((3,3))
    A[0:2,:]=del_tri_pts2.reshape(3,2).T
    if bayesian_coordinates_f.shape[0]!=0:
        
        


        A_coord=A@bayesian_coordinates_f.T
        # print(A_coord.shape)
        # print(bayesian_coordinates_f.shape)
        A_coord=A_coord/A_coord[-1]
        bounded_array_f=bounded_array_f.T
        # print(bounded_array_f[-1])
        bounded_array_f=bounded_array_f/bounded_array_f[-1]
        bounded_array_f=(bounded_array_f).astype(np.int32)
        # print(bounded_array_f.shape)
        A_coord=(A_coord).astype(np.int32)
        temp=img2[bounded_array_f[1,:],bounded_array_f[0,:]].copy()

        img3[bounded_array_f[1,:],bounded_array_f[0,:]] = img2[A_coord[1,:],A_coord[0,:]].copy()
        # img3[A_coord[1,:],A_coord[0,:]]=img2[bounded_array_f[1,:],bounded_array_f[0,:]]
        # img2_black[bounded_array_f[1,:],bounded_array_f[0,:]] = img2[A_coord[1,:],A_coord[0,:]]
        
    
def whole_image_warp_delunay(img1,img2_,del_x,del_y):   
    #A_coord
    img1_black=np.zeros(img2_.shape,dtype=np.uint8)
    img2_black=np.zeros(img2_.shape,dtype=np.uint8)

    # img2_=img1.copy()

    # for i in range(len(delunay_triangle_pts_list[0])):
    #     image_warping(delunay_triangle_pts_list[0][i],delunay_triangle_pts_list[1][i],img1,img2_,img1_black,img2_black)
    for i in range(len(del_x)):
        image_warping(del_x[i],del_y[i],img1,img2_,img1_black,img2_black)
    return img2_,img1_black

def posisson_blending(src,dst,mx,rects):
    poly_left = cv2.convexHull(mx)#img2[op[:,1],op[:,0]]#np.array([(51, 228), (100, 151), (233, 102), (338, 110), (426, 160), (373, 252), (246, 284), (134, 268)], np.int32)

    src_mask = np.zeros((dst.shape), dst.dtype)
    src_mask=cv2.fillPoly(src_mask, [poly_left], (255,255,255))
    # plt.imshow(src_mask)
    im1=cv2.seamlessClone(src,dst,src_mask,(int((rects[1][0]+rects[0][0])/2),int((rects[1][1]+rects[0][1])/2)),cv2.NORMAL_CLONE)
    #im1=cv2.seamlessClone(src,dest,src_mask1,(int((rects[0][1][0]+rects[0][0][0])/2),int((rects[0][1][1]+rects[0][0][1])/2)),cv2.NORMAL_CLONE)
    return im1

from scipy.spatial import distance
def distance_function(pts1,pts1_): #compute r^2xlog(r^2)
    diff_pts=(pts1-pts1_)
    #r_sq=distance.cdist(pts1,pts1_,'sqeuclidean')
    r=np.linalg.norm(diff_pts)
    # r=np.sqrt(r)
    # r_sq=np.square(r)
    r_sq=r
    #print(np.log(r_sq))
    
    U=np.multiply((r_sq**2),np.log(r_sq+1e-5
                               ))    #print(U)
    return U
    

def global_cord_img(x_,y_,h_,w_):
    l1=np.arange(x_,x_+h_+2)
    l2=np.arange(y_,y_+w_+2)
    a=np.meshgrid(l1,l2)
    a=np.array(a)
    #a
    a=np.moveaxis(a,0,-1)
    # print(a.shape)
    ones=np.ones((a.shape[0],a.shape[1],1))
    # print(a[:,1])
    g_cord=np.concatenate([a,ones],-1)
    return g_cord

def distanc_function_forward(patch,reference,weights):
    patch=np.tile(patch[:,:2],(68,1,1,))
    # print(patch.shape)
    _,w,h=patch.shape
    reference=np.tile(np.expand_dims(reference,1),(1,w,1))
    # print(reference.shape)
    diff_pts=(patch-reference)
    
    r=np.linalg.norm(diff_pts,axis=-1)
    
    # r=np.sqrt(r)
    # r_sq=np.square(r)
    r_sq=r
    U=(r_sq**2)*np.log(r_sq+1e-5)
    # print(r.shape)
    w,h=r.shape
    weights=weights[:-3,:]
    #weights=np.expand_dims(weights,-1)
    #weights_exp=np.tile(weights,(1,w,h))
    
    # weights_ex
    summation=U.T@weights
    # print(summation.shape)
    return summation#.shape
def weights_for_tps(mx,my):
    lm_w,lm_h=mx.shape
    k=np.zeros((lm_w,lm_w))
    for i_ in range(len(mx)):
        #for j_ in range(i_,len(multi_img_face_lm[0])):
        for j_ in range(len(mx)):
            k[i_,j_]=distance_function(my[i_],my[j_])


    P=np.ones((lm_w,3))
    P[:,0]=my[:,1]
    P[:,1]=my[:,0]
    A_=np.zeros((lm_w+3,lm_w+3))
    A_[:-3,:-3]=k
    A_[:-3,-3:]=P
    A_[-3:,:-3]=P.T
    I=np.identity(A_.shape[0])
    lamda=1e-3
    A=A_+lamda*I
    A_inv=np.linalg.inv(A)
    x_coord_zeros=np.zeros((A_.shape[0],1))
    x_coord_zeros[:-3,0]=mx[:,0]
    y_coord_zeros=np.zeros((A_.shape[0],1))
    y_coord_zeros[:-3,0]=mx[:,1]



    weights_x=A_inv @ x_coord_zeros
    weights_y=A_inv @ y_coord_zeros
    return weights_x,weights_y
def tps_swaping(mx,my,img2,img3,weights_x,weights_y):
    poly_left = cv2.convexHull(my)#img2[op[:,1],op[:,0]]#np.array([(51, 228), (100, 151), (233, 102), (338, 110), (426, 160), (373, 252), (246, 284), (134, 268)], np.int32)


    src_mask = np.zeros((img3.shape[0],img3.shape[1]), img3.dtype)
    src_mask=cv2.fillPoly(src_mask, [poly_left], (255))
    x,y=np.where(src_mask==255)
    coordinate_x=np.ones((x.shape[0],3))
    coordinate_x[:,0]=x
    coordinate_x[:,1]=y
    # corcy=multi_img_face_lm[0]
    mask=255*np.ones(img2.shape)
    mask=np.uint8(mask)
    # mask.shape

    m,n,_=img2.shape
    poly_left = cv2.convexHull(my)

    #plt.imshow(src_mask)
    corcx=mx.copy()#multi_img_face_lm[0].copy()
    corcx[:,0]=my[:,1].copy()
    corcx[:,1]=my[:,0].copy()

    x,y=np.where(src_mask==255)
    op_x=(coordinate_x @weights_x[-3:,:])+distanc_function_forward(coordinate_x,corcx,weights_x)
    op_y=(coordinate_x @weights_y[-3:,:])+distanc_function_forward(coordinate_x,corcx,weights_y)
    op_x=np.reshape(op_x,(op_x.shape[0]*op_x.shape[1],1))
    op_y=np.reshape(op_y,(op_y.shape[0]*op_y.shape[1],1))

    op=np.concatenate((op_x,op_y),axis=-1)
    # # coordinate_x.shape
    op=np.array(op,np.int32)
    coordinate_x1=np.array(coordinate_x,dtype=np.int32)

    # img3_=np.zeros(img2.shape)
    # img3=img2.copy()
    temp=img3[op[:,1],op[:,0]]


    # img2[op[:,1],op[:,0]]=img3[coordinate_x1[:,0],coordinate_x1[:,1]]
    img2[coordinate_x1[:,0],coordinate_x1[:,1]]=temp
    # img3_[op[:,1],op[:,0]]=img2[coordinate_x1[:,0],coordinate_x1[:,1]]
    return img2
