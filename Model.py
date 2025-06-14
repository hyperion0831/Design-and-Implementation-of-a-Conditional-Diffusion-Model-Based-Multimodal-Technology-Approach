import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		self.edgeDropper = SpAdjDropEdge(args.keepRate)

		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		if audio_embedding != None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		if audio_embedding != None:
			self.audio_embedding = audio_embedding
		else:
			self.audio_embedding = None
		
		self.modal_weight_net = nn.Sequential(
            nn.Linear(args.latdim, 3 if args.data == 'tiktok' else 2),
            nn.Softmax(dim=1)
        )

		self.dropout = nn.Dropout(p=0.1)

		self.leakyrelu = nn.LeakyReLU(0.2)
				
	def getItemEmbeds(self):
		return self.iEmbeds
	
	def getUserEmbeds(self):
		return self.uEmbeds
	
	def getImageFeats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			return image_feats
		else:
			return self.image_trans(self.image_embedding)
	
	def getTextFeats(self):
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			return self.text_trans(self.text_embedding)

	def getAudioFeats(self):
		if self.audio_embedding == None:
			return None
		else:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		return audio_feats

	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)


		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])#第一跳：在“图像相似度图”（image_adj）上做一次邻居聚合
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)#稀疏矩阵×特征矩阵，相当于 GCN 里的一层邻居聚合

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])#将“用户端原始嵌入” 和 “物品的图像特征” 拼到一起
		embedsImage = torch.spmm(adj, embedsImage)#继续在用户–物品交互图（adj）上做一次聚合

		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])#为了做“两跳”聚合，又把上一步聚合后的用户表示
		embedsImage_ = torch.spmm(adj, embedsImage_)#和原始物品嵌入拼起来，再在互动图上多做一次
		embedsImage += embedsImage_#第一跳让用户看物品的特征，第二跳则让物品看到它的“二跳用户”聚合的特征，加到一起，得到完整的“图像模态”节点嵌入
		
		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])#把用户嵌入和物品表示拼到同一个矩阵里，方便一次性用稀疏矩阵乘法 spmm 做消息传递。
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)#图像相似度图的一跳聚合，提取内容信号

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(adj, embedsText)#交互图的一跳聚合，把内容融入协同图

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
		embedsText_ = torch.spmm(adj, embedsText_)#交互图的二跳聚合，加强协同信息
		embedsText += embedsText_

		if audio_adj != None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)
			embedsAudio += embedsAudio_

		embedsImage += args.ris_adj_lambda * embedsImageAdj#残差融合：把邻接和正常卷积结果加权
		embedsText += args.ris_adj_lambda * embedsTextAdj
		if audio_adj != None:
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj
		# ---- 修改开始：使用动态权重融合各模态 ----
		# 先 动态加权融合 各模态表示，得到 embedsModal
		# 计算融合输入：对所有节点取各模态表示的平均
		if audio_adj == None:
			fusion_input = (embedsImage + embedsText) / 2
		else:
			fusion_input = (embedsImage + embedsText + embedsAudio) / 3
		# 使用动态权重网络生成融合权重，输出形状为 [N, num_modals]，N 为节点总数
		dynamic_weight = self.modal_weight_net(fusion_input)
		# 根据是否存在音频模态分别进行加权融合，注意扩展权重维度以便逐元素相乘
		if audio_adj == None:
			embedsModal = dynamic_weight[:, 0:1] * embedsImage + dynamic_weight[:, 1:2] * embedsText
		else:
			embedsModal = (dynamic_weight[:, 0:1] * embedsImage +
			               dynamic_weight[:, 1:2] * embedsText +
			               dynamic_weight[:, 2:3] * embedsAudio)
		# ---- 修改结束 ----
		embeds = embedsModal
		embedsLst = [embeds]
		for gcn in self.gcnLayers:#再把各模态嵌入送入多层 GCN 累加残差：
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)

		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

		return embeds[:args.user], embeds[args.user:]

	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(image_adj, embedsImage)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)

		if audio_adj != None:
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(audio_adj, embedsAudio)

		embeds1 = embedsImage
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1])
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)

		embeds2 = embedsText
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2)

		if audio_adj != None:
			embeds3 = embedsAudio
			embedsLst3 = [embeds3]
			for gcn in self.gcnLayers:
				embeds3 = gcn(adj, embedsLst3[-1])
				embedsLst3.append(embeds3)
			embeds3 = sum(embedsLst3)

		if audio_adj == None:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		else:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]

	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
		
class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims

        # 新增：条件嵌入层，将用户嵌入（维度为 args.latdim）映射到 in_dims_temp[0] 的维度
        self.cond_emb = nn.Linear(args.latdim, in_dims_temp[0])

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True, cond=None):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32).cuda() / (self.time_emb_dim // 2))
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        # 新增：如果传入条件，则将条件经过线性映射后与 h 相加
        if cond is not None:
            h = h + self.cond_emb(cond)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h


class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		max_beta = 0.999
		alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
		betas = []
		for i in range(self.steps):
			t1 = i / self.steps
			t2 = (i + 1) / self.steps
			betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
		return np.array(betas)

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False, cond=None):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t, cond=cond)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t, cond=None):
		model_output = model(x, t, mess_dropout=False, cond=cond)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats, cond=None):# x_start 就是 x₀——未加噪的原始交互向量
		batch_size = x_start.size(0)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()#随机噪声步t
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts, mess_dropout=True, cond=cond)

		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])