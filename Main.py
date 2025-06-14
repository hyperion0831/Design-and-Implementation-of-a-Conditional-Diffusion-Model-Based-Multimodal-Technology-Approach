import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')

		recallMax = 0
		ndcgMax = 0
		precisionMax = 0
		bestEpoch = 0

		log('Model Initialized')

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)

	def prepareModel(self):
		if args.data == 'tiktok':
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).cuda()
		else:
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
		self.opt = torch.optim.Adam(
			self.model.parameters(),
			lr=args.lr, 
			weight_decay=0
		)
		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()# 定义扩散模型（处理噪声生成和扩散过程）
		
		out_dims = eval(args.dims) + [args.item]# 定义去噪模型（针对不同模态）
		in_dims = out_dims[::-1]
		self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr * 0.1, weight_decay=0)
		#GaussianDiffusion 负责生成噪声并控制扩散过程的时间步。
		#Denoise 是去噪网络，每个模态（图像、文本、音频）有独立的去噪模型。
		#输入维度 in_dims 和输出维度 out_dims 是反向的，符合扩散模型的逆向过程

		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
			out_dims = eval(args.dims) + [args.item]
			in_dims = out_dims[::-1]
			self.denoise_model_audio = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
			self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	#用户-物品图重建
	#构建用户-物品矩阵：
	def buildUIMatrix(self, u_list, i_list, edge_list):
		# 构建稀疏矩阵
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)
		# 添加自连接并归一化邻接矩阵

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
		 # 转换为PyTorch稀疏张量
		 # buildUIMatrix 将扩散模型预测的用户-物品交互（通过 p_sample 生成）转换为稀疏邻接矩阵
		 # 归一化邻接矩阵（normalizeAdj）以增强图神经网络的训练稳定性
		 # 每个模态独立生成一个 UI 矩阵（如 image_UI_matrix），用于后续的多模态推荐

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		if args.data == 'tiktok':
			epDiLoss_audio = 0
		steps = trnLoader.dataset.__len__() // args.batch

		diffusionLoader = self.handler.diffusionLoader

		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()#第一次获取用户嵌入：用于训练扩散模型

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			if args.data == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()

			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()
			if args.data == 'tiktok':
				self.denoise_opt_audio.zero_grad()

			# 对每个模态计算扩散损失和对比损失
			diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, cond=uEmbeds[batch_index])
			diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats, cond=uEmbeds[batch_index])
			if args.data == 'tiktok':
				diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats, cond=uEmbeds[batch_index])

			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss #args.e_loss 是对比损失的权重系数
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
			if args.data == 'tiktok':
				loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss

			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()
			if args.data == 'tiktok':
				epDiLoss_audio += loss_audio.item()

			if args.data == 'tiktok':
				loss = loss_image + loss_text + loss_audio
			else:
				loss = loss_image + loss_text

			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.denoise_model_image.parameters(), 1.0)
			torch.nn.utils.clip_grad_norm_(self.denoise_model_text.parameters(), 1.0)
			if args.data == 'tiktok':
				torch.nn.utils.clip_grad_norm_(self.denoise_model_audio.parameters(), 1.0)
			self.denoise_opt_image.step()
			self.denoise_opt_text.step()
			if args.data == 'tiktok':
				self.denoise_opt_audio.step()

			log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

		log('')
		log('Start to re-build UI matrix')

		with torch.no_grad():

			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			if args.data == 'tiktok':
				u_list_audio = []
				i_list_audio = []
				edge_list_audio = []

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
				
				# 第二次获取用户嵌入，用于 p_sample 的 cond
				uEmbeds = self.model.getUserEmbeds().detach()#重新获取最新用户嵌入

				# image
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, args.sampling_steps, args.sampling_noise, cond=uEmbeds[batch_index])
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				# text
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, args.sampling_steps, args.sampling_noise, cond=uEmbeds[batch_index])
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if args.data == 'tiktok':
					# audio
					denoised_batch = self.diffusion_model.p_sample(self.denoise_model_audio, batch_item, args.sampling_steps, args.sampling_noise, cond=uEmbeds[batch_index])
					top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

					for i in range(batch_index.shape[0]):
						for j in range(indices_[i].shape[0]): 
							u_list_audio.append(int(batch_index[i].cpu().numpy()))
							i_list_audio.append(int(indices_[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)

			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)# 对每个模态重建UI矩阵
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			# text
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			if args.data == 'tiktok':
				# audio
				u_list_audio = np.array(u_list_audio)
				i_list_audio = np.array(i_list_audio)
				edge_list_audio = np.array(edge_list_audio)
				self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
				self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

		log('UI matrix built!')

		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			# 通过多模态图生成用户和物品的最终嵌入
			if args.data == 'tiktok':
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			# BPR损失计算
			ancEmbeds = usrEmbeds[ancs]# 用户嵌入
			posEmbeds = itmEmbeds[poss]# 正样本物品嵌入
			negEmbeds = itmEmbeds[negs]# 负样本物品嵌入
			# forward_MM 方法将基础用户-物品图（torchBiAdj）和多模态重建的 UI 矩阵（image_UI_matrix等）结合，通过图神经网络生成最终的用户和物品嵌入
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			# pairPredict 计算用户对正样本和负样本的偏好差异，BPR 损失通过最大化正样本与负样本的差异来优化模型。

			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = self.model.reg_loss() * args.reg
			loss = bprLoss + regLoss
			
			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			# forward_cl_MM 生成不同模态的用户和物品嵌入（例如图像模态和文本模态的嵌入）
			# contrastLoss 函数通过余弦相似度计算对比损失，拉近同一用户/物品在不同模态下的嵌入，推远不同用户/物品的嵌入
			# args.temp 是温度参数，控制对比学习的分布尖锐程度
						# 通过不同模态生成用户和物品的嵌入
			if args.data == 'tiktok':
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			
			# ---- 动态加权对比损失开始 ----
			# 获取用户模态权重（形状：[batch_size, num_modals]）
			user_modal_weights = self.model.modal_weight_net(ancEmbeds)  # ancEmbeds为用户ID嵌入
			
			# ===================== 模态内对比损失 =====================
			# 图像模态 vs 文本模态（用户）
			cl_loss_image_text_usr = contrastLoss(usrEmbeds1, usrEmbeds2, ancs, ancs, args.temp)
			# 图像模态 vs 文本模态（物品）
			cl_loss_image_text_itm = contrastLoss(itmEmbeds1, itmEmbeds2, poss, poss, args.temp)
			
			# 图像模态 vs 音频模态（用户，仅限TikTok）
			if args.data == 'tiktok':
				cl_loss_image_audio_usr = contrastLoss(usrEmbeds1, usrEmbeds3, ancs, ancs, args.temp)
				cl_loss_image_audio_itm = contrastLoss(itmEmbeds1, itmEmbeds3, poss, poss, args.temp)
				# 文本模态 vs 音频模态（用户）
				cl_loss_text_audio_usr = contrastLoss(usrEmbeds2, usrEmbeds3, ancs, ancs, args.temp)
				cl_loss_text_audio_itm = contrastLoss(itmEmbeds2, itmEmbeds3, poss, poss, args.temp)
			
			# ===================== 新增：用户-物品跨模态对比 =====================
			# 图像模态下用户与物品对齐
			cl_loss_usr_itm_image = contrastLoss(usrEmbeds1, itmEmbeds1, ancs, poss, args.temp)
			# 文本模态下用户与物品对齐
			cl_loss_usr_itm_text = contrastLoss(usrEmbeds2, itmEmbeds2, ancs, poss, args.temp)
			# 音频模态下用户与物品对齐（仅限TikTok）
			if args.data == 'tiktok':
				cl_loss_usr_itm_audio = contrastLoss(usrEmbeds3, itmEmbeds3, ancs, poss, args.temp)
			
			# ===================== 动态加权总损失 =====================
			if args.data == 'tiktok':
				clLoss = (
					# 模态间对比（用户+物品）
					user_modal_weights[:, 0] * user_modal_weights[:, 1] * (cl_loss_image_text_usr + cl_loss_image_text_itm) +
					user_modal_weights[:, 0] * user_modal_weights[:, 2] * (cl_loss_image_audio_usr + cl_loss_image_audio_itm) +
					user_modal_weights[:, 1] * user_modal_weights[:, 2] * (cl_loss_text_audio_usr + cl_loss_text_audio_itm) +
					# 跨模态用户-物品对比
					user_modal_weights[:, 0] * cl_loss_usr_itm_image +
					user_modal_weights[:, 1] * cl_loss_usr_itm_text +
					user_modal_weights[:, 2] * cl_loss_usr_itm_audio
				).mean() * args.ssl_reg
			else:
				clLoss = (
					user_modal_weights[:, 0] * user_modal_weights[:, 1] * (cl_loss_image_text_usr + cl_loss_image_text_itm) +
					user_modal_weights[:, 0] * cl_loss_usr_itm_image +
					user_modal_weights[:, 1] * cl_loss_usr_itm_text
				).mean() * args.ssl_reg
			# ---- 动态加权对比损失结束 ----

			loss += clLoss

			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()# 更新 self.uEmbeds

			log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i, 
				steps,
				bprLoss.item(),
        regLoss.item(),
				clLoss.item()
				), save=False, oneline=True)

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
		ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
		if args.data == 'tiktok':
			ret['Di audio loss'] = epDiLoss_audio / (diffusionLoader.dataset.__len__() // args.batch)
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat

		if args.data == 'tiktok':
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
		else:
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, args.topk)
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allPrecision = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = precision = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
					precision += 1
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			precision = precision / args.topk
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
		return allRecall, allNdcg, allPrecision

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()