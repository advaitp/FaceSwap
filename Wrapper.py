import argparse
import cv2
import numpy as np
import dlib
import scipy
import cvlib as cv
import matplotlib.pyplot as plt
from utilities import * 
from test3 import detectFace
from api import PRN

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--method', default="PRNet", help='type of Faceswapper:DelTri, TPspline,PRNet')
	Parser.add_argument('--DataPath', default="./TestSet_P2/", help='base path where data files exist')
	Parser.add_argument('--VideoName', default="Test2.mp4", help='Video Name')
	Parser.add_argument('--RefImageName', default='None', help=' Reference Image')
	Parser.add_argument('--SavePath', default="./Results/", help='Folder to save results')
	Parser.add_argument('--LandmarkPath', default='./shape_predictor_68_face_landmarks.dat', help= 'dlib shape predictor path')
	Parser.add_argument('--DlibPath', default='./mmod_human_face_detector.dat', help= 'dlib face predictor path')
	

	Args = Parser.parse_args("")
	DataPath = Args.DataPath
	RefImageName = Args.RefImageName
	SavePath = Args.SavePath
	method = Args.method
	VideoName = Args.VideoName
	LM_path = Args.LandmarkPath
	RefImageFilePath = DataPath + RefImageName
	VideoFilePath = DataPath + VideoName

	print(method)
	 
	# SaveFileName = DataPath + SavePath 
	SaveFileName = SavePath +method+ VideoName
	print('Reading ref image.......')
	FaceRef = cv2.imread(RefImageFilePath) ## color image
 
	if FaceRef is None:
		mode = 2
		print('2 face video')
	else:
		mode = 1
		print('1 face video')
	import matplotlib.pyplot as plt
	cap=cv2.VideoCapture(VideoFilePath)
	for i in range(1):
		_,im=cap.read()
	m,n,_=im.shape
	out=cv2.VideoWriter(SaveFileName,cv2.VideoWriter_fourcc(*'mp4v'), 30, (n,m))
	if mode==2:
		if method!='PRNet':
			ret,img1=cap.read()
			img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
			no_of_faces,hog_predictor,rects=detect_face_cnn(img1_gray,img1,LM_path,mode)
			tracker = cv2.TrackerMIL_create()
			tracker1 = cv2.TrackerMIL_create()
			prev_rects=rects
			# tracker = cv2.TrackerMIL_create()
			# tracker1 = cv2.TrackerMIL_create()
			ok = tracker.init(img1, (rects[0].left(),rects[0].top(),rects[0].right()-rects[0].left(),rects[0].bottom()-rects[0].top()))
			ok=tracker1.init(img1,(rects[1].left(),rects[1].top(),rects[1].right()-rects[1].left(),rects[1].bottom()-rects[1].top()))
			# print((rects[1].left(),rects[1].top(),rects[1].right()-rects[1].left(),rects[1].bottom()-rects[1].top()))
			while cap.isOpened():
				print(1)
				ret,img1=cap.read()
				
				if ret==False:
					break
				img=img1.copy()
				img2=img1.copy()
				img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
				no_of_faces,hog_predictor,rects=detect_face_cnn(img1_gray,img1,LM_path,mode)
				tracker.update(img1_gray)
				tracker1.update(img1_gray)
				# no_of_faces,hog_predictor,rects,tracker,tracker1=detect_face(img1_gray,mode,LM_path)
				if no_of_faces!=2  :
					
					rects=prev_rects
					# ok = tracker.init(img1_gray, (rects[0].left(),rects[0].top(),rects[0].right()-rects[0].left(),rects[0].bottom()-rects[0].top()))
					# ok = tracker1.init(img1_gray,(rects[1].left(),rects[1].top(),rects[1].right()-rects[1].left(),rects[1].bottom()-rects[1].top()))
					no_of_faces,hog_predictor,bbox=track_face(img1_gray,tracker,tracker1,mode)
					rects=bbox
#                 prev_rects=rects
#                 if rects[0].left()-prev_rects[0].left()>30 or rects[0].left()-prev_rects[0].left()>30 or rects[1].left()-prev_rects[1].left()>30 or rects[1].left()-prev_rects[1].left()>30  :
					
#                     rects=prev_rects
#                     ok = tracker.init(img1_gray, (rects[0].left(),rects[0].top(),rects[0].right()-rects[0].left(),rects[0].bottom()-rects[0].top()))
#                     ok = tracker1.init(img1_gray,(rects[1].left(),rects[1].top(),rects[1].right()-rects[1].left(),rects[1].bottom()-rects[1].top()))
#                     no_of_faces,hog_predictor,bbox=track_face(img1_gray,tracker,tracker1,mode)
				prev_rects=rects
				# print(no_of_faces)
				#and no_of_faces==2:
				subdiv2=[]
				bounding_box=[]
				multi_img_face_lm=[]

				landmarks_index_list=[]
				rect_=[]

				# return bounding,landmarks_index,facial_landmarks
				bounding,landmarks_index,facial_landmarks=landmarks_extraction(img1_gray,hog_predictor,rects[0])
				bounding_box.append(bounding)
				multi_img_face_lm.append(facial_landmarks)
				landmarks_index_list.append(landmarks_index)

				bounding,landmarks_index,facial_landmarks=landmarks_extraction(img1_gray,hog_predictor,rects[1])
				bounding_box.append(bounding)
				multi_img_face_lm.append(facial_landmarks)
				landmarks_index_list.append(landmarks_index)

				rect1,subdiv1=(get_bounding_box(multi_img_face_lm[0]))
				rect_.append(rect1)
				subdiv2.append(subdiv1)

				rect2,subdiv1=(get_bounding_box(multi_img_face_lm[1]))
				rect_.append(rect2)
				subdiv2.append(subdiv1)
				# print(mode)
				if method=='DelTri': 
					delunay_traingle_pts_list=delunay_triangle(subdiv2,multi_img_face_lm)
					img_with_delunay_pts,sorted_delunay_lists,sorting_index_for_delunay_list=matching_delunay_triangles(delunay_traingle_pts_list,rect_,landmarks_index_list,img1.copy(),multi_img_face_lm)
					delunay_triangle_pts_list=sorted_delunay_lists
					img2_=img1.copy()
					img1_black=np.zeros(img2.shape,dtype=np.uint8)
					img2_black=np.zeros(img2.shape,dtype=np.uint8)

					im1,im1b=whole_image_warp_delunay(img1.copy(),img1.copy(),delunay_triangle_pts_list[0].copy(),delunay_triangle_pts_list[1].copy())
					im1,im1b=whole_image_warp_delunay(img1.copy(),im1,delunay_triangle_pts_list[1].copy(),delunay_triangle_pts_list[0].copy())
					im=im1

					rects=rect_
					# print(rects.shape)
					im1=posisson_blending(im,img2,multi_img_face_lm[1],rects[1])
					im1=posisson_blending(im,im1,multi_img_face_lm[0],rects[0])
					cv2.imshow('Frame',im1)
					#cv2.imwrite("im1_d.jpg",im1)
					#plt.imshow(im1)
					key = cv2.waitKey(1)
					# print(i_)
					# i_=i_+1
					out.write(im1)
					# print('3')
					if key == ord('q'):

					  break
				elif method=='TPspline':
					rects=rect_
					weights_x,weights_y=weights_for_tps(multi_img_face_lm[0],multi_img_face_lm[1])
					im=tps_swaping(multi_img_face_lm[0].copy(),multi_img_face_lm[1].copy(),img2.copy(),img2.copy(),weights_x,weights_y)


					weights_x,weights_y=weights_for_tps(multi_img_face_lm[1],multi_img_face_lm[0])
					im=tps_swaping(multi_img_face_lm[1].copy(),multi_img_face_lm[0].copy(),im.copy(),img2.copy(),weights_x,weights_y)


					im1=posisson_blending(im,img2,multi_img_face_lm[1],rects[1])
					im1=posisson_blending(im,im1,multi_img_face_lm[0],rects[0])
					cv2.imshow('Frame',im1)

					# 20 is in milliseconds, try to increase the value, say 50 and observe
					key = cv2.waitKey(1)

					out.write(im1)
					if key == ord('q'):

					  break
			
			cap.release()

			out.release()
			cv2.destroyAllWindows()
		
		else : 
			while cap.isOpened():
				try : 
					ret, frame = cap.read()
					prn = PRN(is_dlib = True)
					img = detectFace(frame, prn, mode)
					if i == 0 :
						cv2.imwrite(f'scarletts{i}.jpg', img)
						
					print(f'Running for frame : {i+1}')
					i += 1
					vid.write(img)
				except Exception as e :
					print(f'Exception occured neglecting frame : {e}')
					pass

			out.release()
			cv2.destroyAllWindows()


	if mode==1:
		if method!='PRNet':
			re_=1
			rects=[]
			rect1=[]
			ret,img1=cap.read()
			while len(rect1)==0:
				ret,img1=cap.read()
				img2=FaceRef

				img1 = cv2.resize(img1,(480,854))
				img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
				no_of_faces,hog_predictor,rect1=detect_face_cnn(img1_gray,img1,LM_path,mode)
				img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
				
				no_of_faces,hog_predictor2,rect2_=detect_face_cnn(img2_gray,img2,LM_path,mode)
			rects.append(rect1[0])
			rects.append(rect2_[0])
			# print(rect1)
			tracker = cv2.TrackerMIL_create()
			tracker1 = cv2.TrackerMIL_create()
			prev_rects=rects

			ok = tracker.init(img1, (rects[0].left(),rects[0].top(),rects[0].right()-rects[0].left(),rects[0].bottom()-rects[0].top()))
			# ok=tracker1.init(img1,(rects[1].left(),rects[1].top(),rects[1].right()-rects[1].left(),rects[1].bottom()-rects[1].top()))
			# print((rects[1].left(),rects[1].top(),rects[1].right()-rects[1].left(),rects[1].bottom()-rects[1].top()))

			while cap.isOpened():
				# print(1)
				ret,img1=cap.read()
				
				if ret==False:
					break
				img1 = cv2.resize(img1,(480,854))
				img=img1.copy()
				
				
				img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
				no_of_faces_,hog_predictor,rects=detect_face_cnn(img1_gray,img1,LM_path,mode)
				cv2.imshow("img2",img2_gray)
				no_of_faces,hog_predictor2,rect2_=detect_face_cnn(img2_gray,img2,LM_path,mode)
				rects.append(rect2_[0])
				
				# rects=re
				# no_of_faces,hog_predictor,rects,tracker,tracker1=detect_face(img1_gray,mode,LM_path)
				if no_of_faces_==1:
					ok=tracker.init(img1_gray,(rects[0].left(),rects[0].top(),rects[0].right()-rects[0].left(),rects[0].bottom()-rects[0].top()))
					tracker.update(img1_gray)
				if no_of_faces_!=1  :
					
					rects=prev_rects
					# ok = tracker.init(img1_gray, (rects[0].left(),rects[0].top(),rects[0].right()-rects[0].left(),rects[0].bottom()-rects[0].top()))
					# ok = tracker1.init(img1_gray,(rects[1].left(),rects[1].top(),rects[1].right()-rects[1].left(),rects[1].bottom()-rects[1].top()))
					no_of_faces,hog_predictor,bbox=track_face(img1_gray,tracker,tracker,mode)
					rects=bbox
					rects.append(rect2_[0])

#                 prev_rects=rects
#                 if rects[0].left()-prev_rects[0].left()>30 or rects[0].left()-prev_rects[0].left()>30 or rects[1].left()-prev_rects[1].left()>30 or rects[1].left()-prev_rects[1].left()>30  :
				 
#                     rects=prev_rects
#                     ok = tracker.init(img1_gray, (rects[0].left(),rects[0].top(),rects[0].right()-rects[0].left(),rects[0].bottom()-rects[0].top()))
#                     ok = tracker1.init(img1_gray,(rects[1].left(),rects[1].top(),rects[1].right()-rects[1].left(),rects[1].bottom()-rects[1].top()))
#                     no_of_faces,hog_predictor,bbox=track_face(img1_gray,tracker,tracker1,mode)
				prev_rects=rects
				# print(no_of_faces)
				#and no_of_faces==2:
				subdiv2=[]
				bounding_box=[]
				multi_img_face_lm=[]

				landmarks_index_list=[]
				rect_=[]
				print(rects)
				# return bounding,landmarks_index,facial_landmarks
				bounding,landmarks_index,facial_landmarks=landmarks_extraction(img1_gray,hog_predictor,rects[0])
				bounding_box.append(bounding)
				multi_img_face_lm.append(facial_landmarks)
				landmarks_index_list.append(landmarks_index)
				# print('1',rects[0],rects[1])
				bounding,landmarks_index,facial_landmarks=landmarks_extraction(img2_gray,hog_predictor2,rects[1])
				bounding_box.append(bounding)
				multi_img_face_lm.append(facial_landmarks)
				landmarks_index_list.append(landmarks_index)
				
				rect1,subdiv1=(get_bounding_box(multi_img_face_lm[0]))
				rect_.append(rect1)
				subdiv2.append(subdiv1)

				rect2,subdiv1=(get_bounding_box(multi_img_face_lm[1]))
				rect_.append(rect2)
				subdiv2.append(subdiv1)
				# print(mode)
				if method=='DelTri': 
					delunay_traingle_pts_list=delunay_triangle(subdiv2,multi_img_face_lm)
					img_with_delunay_pts,sorted_delunay_lists,sorting_index_for_delunay_list=matching_delunay_triangles(delunay_traingle_pts_list,rect_,landmarks_index_list,img1.copy(),multi_img_face_lm)
					delunay_triangle_pts_list=sorted_delunay_lists
					# img2_=img1.copy()
					img1_black=np.zeros(img2.shape,dtype=np.uint8)
					img2_black=np.zeros(img2.shape,dtype=np.uint8)
					
					im1,im1b=whole_image_warp_delunay(img2.copy(),img1.copy(),delunay_triangle_pts_list[0].copy(),delunay_triangle_pts_list[1].copy())
					# im1,im1b=whole_image_warp_delunay(img1.copy(),im1,delunay_triangle_pts_list[1].copy(),delunay_triangle_pts_list[0].copy())
					im=im1

					rects=rect_
					# print(rects.shape)
					im1=posisson_blending(im,img1,multi_img_face_lm[0],rects[0])
					# im1=posisson_blending(im,im1,multi_img_face_lm[0],rects[0])
					cv2.imshow('Frame',im1)
					#cv2.imwrite("im1_d.jpg",im1)
					#plt.imshow(im1)
					key = cv2.waitKey(1)
					# print(i_)
					# i_=i_+1
					out.write(im1)
					# print('3')
					if key == ord('q'):

					  break
				elif method=='TPspline':
					rects=rect_
					weights_x,weights_y=weights_for_tps(multi_img_face_lm[1],multi_img_face_lm[0])
					im=tps_swaping(multi_img_face_lm[1].copy(),multi_img_face_lm[0].copy(),img1.copy(),img2.copy(),weights_x,weights_y)


					# weights_x,weights_y=weights_for_tps(multi_img_face_lm[1],multi_img_face_lm[0])
					# im=tps_swaping(multi_img_face_lm[1].copy(),multi_img_face_lm[0].copy(),im.copy(),img2.copy(),weights_x,weights_y)


					# im1=posisson_blending(im,img1,multi_img_face_lm[1],rects[1])
					im1=posisson_blending(im,img1,multi_img_face_lm[0],rects[0])
					cv2.imshow('Frame',im1)

					# 20 is in milliseconds, try to increase the value, say 50 and observe
					key = cv2.waitKey(1)

					out.write(im1)
					if key == ord('q'):

					  break
			
			cap.release()

			out.release()
			cv2.destroyAllWindows()		
		else : 
			while cap.isOpened():
				try : 
					ret, frame = cap.read()
					prn = PRN(is_dlib = True)
					img = detectFace(frame, prn, mode)
					if i == 0 :
						cv2.imwrite(f'scarletts{i}.jpg', img)
						
					print(f'Running for frame : {i+1}')
					i += 1
					vid.write(img)
				except Exception as e :
					print(f'Exception occured neglecting frame : {e}')
					pass

			out.release()

			cv2.destroyAllWindows()
if __name__ =='__main__':
	main()