import os
import cv2
import numpy as np
import sys
import pickle
import time
from keras_frcnn import config
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

class Predictor:

	def __init__(self):
		sys.setrecursionlimit(40000)

		config_output_filename = 'config.pickle'
		with open(config_output_filename, 'r') as f_in:
			self.C = pickle.load(f_in)
			
		# turn off any data augmentation at test time
		self.C.use_horizontal_flips = False
		self.C.use_vertical_flips = False
		self.C.rot_90 = False

		self.build_models()

	def format_img(self, img, C):
		img_min_side = float(C.im_size)
		(height,width,_) = img.shape
		
		if width <= height:
			f = img_min_side/width
			new_height = int(f * height)
			new_width = int(img_min_side)
		else:
			f = img_min_side/height
			new_width = int(f * width)
			new_height = int(img_min_side)
		img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
		img = img[:, :, (2, 1, 0)]
		img = img.astype(np.float32)
		img[:, :, 0] -= C.img_channel_mean[0]
		img[:, :, 1] -= C.img_channel_mean[1]
		img[:, :, 2] -= C.img_channel_mean[2]
		img /= C.img_scaling_factor
		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)
		return img

	def build_models(self):
		self.class_mapping = self.C.class_mapping

		if 'bg' not in self.class_mapping:
			self.class_mapping['bg'] = len(self.class_mapping)

		self.class_mapping = {v: k for k, v in self.class_mapping.iteritems()}
		self.C.num_rois = 32

		if K.image_dim_ordering() == 'th':
			input_shape_img = (3, None, None)
			input_shape_features = (1024, None, None)
		else:
			input_shape_img = (None, None, 3)
			input_shape_features = (None, None, 1024)


		img_input = Input(shape=input_shape_img)
		roi_input = Input(shape=(self.C.num_rois, 4))
		feature_map_input = Input(shape=input_shape_features)

		# define the base network (resnet here, can be VGG, Inception, etc)
		shared_layers = nn.nn_base(img_input, trainable=True)

		# define the RPN, built on the base layers
		num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
		rpn_layers = nn.rpn(shared_layers, num_anchors)

		classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.class_mapping), trainable=True)

		self.model_rpn = Model(img_input, rpn_layers)
		self.model_classifier_only = Model([feature_map_input, roi_input], classifier)

		model_classifier = Model([feature_map_input, roi_input], classifier)

		self.model_rpn.load_weights(self.C.model_path, by_name=True)
		model_classifier.load_weights(self.C.model_path, by_name=True)

		self.bbox_threshold = 0.8



	def predict(self, img=None, filepath=None, img_name=None):
		if img is None:
			img = cv2.imread(filepath)

		X = self.format_img(img, self.C)

		img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
		img_scaled[:, :, 0] += 123.68
		img_scaled[:, :, 1] += 116.779
		img_scaled[:, :, 2] += 103.939
		
		img_scaled = img_scaled.astype(np.uint8)

		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2, F] = self.model_rpn.predict(X)
		

		R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)

		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}

		for jk in range(R.shape[0]//self.C.num_rois + 1):
			ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0]//self.C.num_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],self.C.num_rois,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

			for ii in range(P_cls.shape[1]):

				if np.max(P_cls[0, ii, :]) < self.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
					tx /= self.C.classifier_regr_std[0]
					ty /= self.C.classifier_regr_std[1]
					tw /= self.C.classifier_regr_std[2]
					th /= self.C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([16*x, 16*y, 16*(x+w), 16*(y+h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		all_dets = []

		for key in bboxes:
			bbox = np.array(bboxes[key])

			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]

				cv2.rectangle(img_scaled,(x1, y1), (x2, y2), np.array([0, 255, 255]), 2)

				all_dets.append([new_probs[jk], x1, y1, x2, y2])

		# cv2.imshow('img', img_scaled)
		# cv2.waitKey(0)
		# cv2.imwrite('./imgs/{}.png'.format(img_name[:-4]),img_scaled)
		return all_dets


def main():
	p = Predictor()
	p.predict(filepath='frame300crop.jpg', img_name='frame1.jpg')

if __name__ == '__main__':
	main()
