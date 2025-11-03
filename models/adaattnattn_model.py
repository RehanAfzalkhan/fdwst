import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks
import matplotlib.pyplot as plt
from .verifier import Verifier
from data.wavelets import *
import os.path, cv2
from glob import glob
from mate_models.networks.dynast_transformer import DynamicTransformerBlock
from mate_models.networks.generator import DynaSTGenerator

class AdaAttNattnModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--image_encoder_path', required=True,       help='path to pretrained image encoder')
        parser.add_argument('--skip_connection_3',  action='store_true', help='if specified, add skip connection on ReLU-3')
        parser.add_argument('--shallow_layer',      action='store_true', help='if specified, also use features of shallow layers')
        if is_train:
            parser.add_argument('--lambda_content', type=float, default=0.,  help='weight for L2 content loss')
            parser.add_argument('--lambda_global',  type=float, default=10., help='weight for L2 style loss')
            parser.add_argument('--lambda_local',   type=float, default=3.,  help='weight for attention weighted style loss')
            parser.add_argument('--lambda_blur',    type=float, default=5.,  help='weight for gamma blur score loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        image_encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )

        gamma_linear_encoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

        attn_enc = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

        embed_dim_qk = 128
        embed_dim_v = 64
        dim_prune = 64
        ic = 64

        self.pos_embed = nn.Parameter(torch.randn(1, 64, 32, 32)).to(self.device)
        mate_gen = DynamicTransformerBlock(embed_dim_qk, embed_dim_v, dim_prune, ic, smooth = None)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, 0.0005)
                m.bias.data.fill_(0.01)

        self.gamma = nn.Parameter(torch.rand((64),  requires_grad = True, device = opt.gpu_ids[0])) # multiple random inits
        # self.gamma = nn.Parameter(torch.full(size = (64, 1), fill_value = 0.9, requires_grad = True, device = opt.gpu_ids[0])) # multiple set to a value
        # self.gamma = nn.Parameter(torch.tensor(0.9, requires_grad = True, device = opt.gpu_ids[0])) # single set to a value

        image_encoder.load_state_dict(torch.load(opt.image_encoder_path))
        enc_layers = list(image_encoder.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[  : 4]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[4 :11]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[11:18]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[18:31]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[31:44]).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]

        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.net_gamma_enc = nn.DataParallel(nn.Sequential(image_encoder).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.net_gamma_lin = nn.DataParallel(nn.Sequential(gamma_linear_encoder).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.net_enc_q     = nn.DataParallel(nn.Sequential(attn_enc).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.net_enc_k     = nn.DataParallel(nn.Sequential(attn_enc).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.net_enc_v     = nn.DataParallel(nn.Sequential(attn_enc).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.net_matebit_generator = mate_gen.to(self.device)

        for param in self.net_gamma_enc.parameters()     : param.requires_grad = True
        for param in self.net_gamma_lin.parameters()     : param.requires_grad = True
        for param in self.net_enc_q.parameters()         : param.requires_grad = True
        for param in self.net_enc_k.parameters()         : param.requires_grad = True
        for param in self.net_enc_v.parameters()         : param.requires_grad = True
        # for param in self.net_matebit_generator.modules.parameters() : param.requires_grad = True

        self.visual_names = ['blur', 'cs', 's']
        self.model_names =  ['decoder', 'gamma_enc', 'gamma_lin', 'enc_q', 'enc_k', 'enc_v', 'matebit_generator'] #, 'transformer']
        parameters = []
        self.max_sample = 64 * 64
        if opt.skip_connection_3:
            adaattn_3 = networks.AdaAttN(in_planes=256, key_planes = 256 + 128 + 64 if opt.shallow_layer else 256, max_sample=self.max_sample)
            self.net_adaattn_3 = networks.init_net(adaattn_3, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.net_adaattn_3.module.load_state_dict(torch.load("/home/admin/AdaAttN/checkpoints/AdaAttN_NEW_BIN_SAME_CS/latest_net_adaattn_3.pth"))
            self.model_names.append('adaattn_3')
            parameters.append(self.net_adaattn_3.parameters())
        if opt.shallow_layer : channels = 512 + 256 + 128 + 64
        else : channels = 512

        decoder = networks.Decoder(opt.skip_connection_3)
        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_decoder.module.load_state_dict(torch.load("/home/admin/AdaAttN/checkpoints/AdaAttN_NEW_BIN_SAME_CS/latest_net_decoder.pth"), strict = False)

        # self.net_gamma_lin.apply(init_weights)

        # self.net_gamma_enc.module.load_state_dict(torch.load("/home/admin/AdaAttN/checkpoints/AdaAttN_real_fixed_blur_gamma/latest_net_gamma_enc.pth"), strict = False)
        self.net_gamma_lin.module.load_state_dict(torch.load("/home/admin/AdaAttN/checkpoints/AdaAttN_real_fixed_blur_gamma/latest_net_gamma_lin.pth"), strict = False)
        # self.net_enc_q.module.load_state_dict(torch.load("/home/admin/AdaAttN/checkpoints/AdaAttN_real_fixed_blur_gamma/latest_net_gamma_lin.pth"), strict = False)
        # self.net_enc_k.module.load_state_dict(torch.load("/home/admin/AdaAttN/checkpoints/AdaAttN_real_fixed_blur_gamma/latest_net_gamma_lin.pth"), strict = False)
        # self.net_enc_v.module.load_state_dict(torch.load("/home/admin/AdaAttN/checkpoints/AdaAttN_real_fixed_blur_gamma/latest_net_gamma_lin.pth"), strict = False)

        # transformer = networks.Transformer(in_planes=512, key_planes=channels, shallow_layer=opt.shallow_layer)
        # self.net_transformer = networks.init_net(transformer, opt.init_type, opt.init_gain, opt.gpu_ids)
        # self.net_transformer.module.load_state_dict(torch.load("/home/admin/AdaAttN/checkpoints/AdaAttN/latest_net_transformer.pth"))
        # parameters.append(self.net_transformer.parameters())

        verif_weights = torch.load('/home/admin/AdaAttN/models/model_resnet18_100.pt')
        self.verif_photo = Verifier().to(self.device)
        self.verif_photo.load_state_dict(verif_weights['net_photo'])
        self.verif_photo = nn.DataParallel(Verifier().to(opt.gpu_ids[0]), opt.gpu_ids)
        self.verif_photo.eval()
        for param in self.verif_photo.parameters():
            param.requires_grad = False
        print('Verifier initialized...')

        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_gamma_enc.parameters())
        parameters.append(self.net_gamma_lin.parameters())
        parameters.append(self.net_enc_q.parameters())
        parameters.append(self.net_enc_k.parameters())
        parameters.append(self.net_enc_v.parameters())
        parameters.append(self.net_matebit_generator.parameters())


        def gamma_gen() : yield self.gamma
        parameters.append(gamma_gen())

        self.c          = None
        self.cs         = None
        self.s          = None
        self.blur       = None
        self.bin        = None
        self.name_A     = None
        self.name_B     = None
        self.s_feats    = None
        self.c_feats    = None
        self.b_feats    = None
        self.gamma_blur = None
        self.blur_score = None
        self.seed       = 6666

        if self.isTrain:
            self.loss_names    = ['content', 'global', 'local', 'identity', 'blur']
            self.criterionMSE  = torch.nn.MSELoss().to(self.device)
            self.L1            = torch.nn.L1Loss().to(self.device)
            self.optimizer_g   = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_global   = torch.tensor(0., device=self.device)
            self.loss_identity = torch.tensor(0., device=self.device)
            self.loss_local    = torch.tensor(0., device=self.device)
            self.loss_content  = torch.tensor(0., device=self.device)
            self.loss_blur     = torch.tensor(0., device=self.device)

    def set_input(self, input_dict):
        self.s       = input_dict['sharp'].to(self.device)
        self.blur    = input_dict['blur'].to(self.device)
        self.bin     = input_dict['bin'].to(self.device)
        self.name_A  = input_dict['Name_A']
        self.name_B  = input_dict['Name_B']
        self.blur_score = input_dict['blur_scores'].to(self.device)
        self.image_paths = input_dict['name']

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]
    
    def encode_with_gamma(self, input_img):
        return self.net_gamma_enc(input_img)
    
    def encode_with_q(self, input_img):
        return self.net_enc_q(input_img)
    
    def encode_with_k(self, input_img):
        return self.net_enc_k(input_img)
    
    def encode_with_v(self, input_img):
        return self.net_enc_v(input_img)
    
    def encode_with_gamma_lin(self, input):
        input = torch.mean(input, [2, 3])
        # print(input.shape)
        return self.net_gamma_lin(input)

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(networks.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(networks.mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return networks.mean_variance_norm(feats[last_layer_idx])

    def forward(self):

        G_waves = []
        T_waves = []
        B_waves = []
        T_img = []
        B_img = []

        for img, bin_img in zip(self.blur, self.bin):
            T_w, B_w, T, B = get_wavelets(torch.squeeze(img.cpu()), torch.squeeze(bin_img.cpu()), self.name_A, self.name_B)
            T_waves.append(T_w)
            B_waves.append(B_w)
            T_img.append(T)
            B_img.append(B)

        B_img_t = torch.tensor(B_img, device = self.device)
        B_img_t = B_img_t[:, None, :, :]
        B_img_t = B_img_t.repeat(1, 3, 1, 1)
        B_waves_t = torch.stack(B_waves).to(self.device)
        T_waves_t = torch.stack(T_waves).to(self.device)

        
        Q = self.encode_with_q(B_waves_t.permute(0, 2, 3, 1))
        K = self.encode_with_k(T_waves_t.permute(0, 2, 3, 1))
        V = K

        pos = self.pos_embed

        y, output2, map = self.net_matebit_generator(Q, K, V, pos, None)

        # print(map.shape)

        # fig = plt.figure(figsize = (15, 9), layout = "tight")

        # ax = fig.add_subplot(2, 2, 1)
        # ax.imshow(Q[0, 0].cpu().detach(), cmap = "gray") # Blurry image
        # ax.set_title("q")

        # ax = fig.add_subplot(2, 2, 2)
        # ax.imshow(y[0, 0].cpu().detach(), cmap = "gray") # Blurry image
        # ax.set_title("y")

        # ax = fig.add_subplot(2, 2, 3)
        # ax.imshow(map[0].cpu().detach(), cmap = "gray") # Blurry image
        # ax.set_title("map")

        # plt.show()

        self.B_img_feats = self.encode_with_gamma(B_img_t)
        self.B_img_feats = self.encode_with_gamma_lin(self.B_img_feats)
        self.gamma_blur = torch.squeeze(self.B_img_feats)

        ######

        # print(y.shape)
        # print(T_waves_t.shape)
        T_waves_t = y.permute(0, 3, 1, 2)
        
        ######

        for i, (T, B) in enumerate(zip(T_waves_t.permute(0, 2, 3, 1), B_waves_t.permute(0, 2, 3, 1))):
            G_waves.append((self.gamma[i] * B.to(self.device)) + ((1.0 - self.gamma[i]) * T.to(self.device))) # old
            # G_waves.append(((self.gamma[i] + self.gamma_blur.item() / 2423.05) * B.to(self.device)) + ((1.0 - self.gamma[i]) * T.to(self.device))) # new

        G_waves = torch.stack(G_waves, dim = 0).permute(0, 3, 1, 2)

        if len(G_waves.shape) == 3 : self.c = torch.unsqueeze(G_waves.permute(1, 2, 0).to(self.device), 0)
        else                       : self.c = G_waves.permute(0, 2, 3, 1)

        self.s_feats = self.encode_with_intermediate(self.s)
        self.cs = self.net_decoder(self.c)

    def compute_identity_loss(self):
        real_embedding, real_features = self.verif_photo(self.s)
        stylized_embedding, stylized_features = self.verif_photo(self.cs)
        loss_verif_features = 0.0
        lambda_ridge_features = [1.0, 1.0, 1.0]
        for i in range(3):
            f_real = real_features[i]
            f_st = stylized_features[i]
            loss_verif_features += self.criterionMSE(f_st, f_real) * lambda_ridge_features[i]
        self.loss_identity = self.criterionMSE(stylized_embedding, real_embedding) * 25.0 + loss_verif_features

    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.opt.lambda_content > 0:
            # if self.s.shape != self.cs.shape : 
            #     self.cs = self.cs[:,:,:-4,:-4] # FOR DIFFERNET SIZES
            self.loss_content += self.L1(self.cs, self.s) ############# c_feats

    def compute_style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)
        if self.opt.lambda_global > 0:
            for i in range(1, 5):
                s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
                self.loss_global += self.criterionMSE(
                    stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)

        self.loss_local = torch.tensor(0., device=self.device)
        if self.opt.lambda_local > 0:
            for i in range(1, 5):
                c_key = self.get_key(self.c_feats, i, self.opt.shallow_layer)
                s_key = self.get_key(self.s_feats, i, self.opt.shallow_layer)
                s_value = self.s_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                if h_s * w_s > self.max_sample:
                    torch.manual_seed(self.seed)
                    index = torch.randperm(h_s * w_s).to(self.device)[:self.max_sample]
                    s_key = s_key[:, :, index]
                    style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
                else:
                    style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                self.loss_local += self.criterionMSE(stylized_feats[i], std * networks.mean_variance_norm(self.c_feats[i]) + mean)

    def compute_blur_loss(self):
        # print("score: ", self.blur_score)
        # print("gamma: ", self.gamma_blur)
        # if self.blur_score.shape
        # self.loss_blur = self.blur_score - self.gamma_blur
        # print(self.gamma_blur.dtype)
        # print(self.blur_score.dtype)
        flag = False
        for s in self.gamma_blur:
            if s == 0.0000 : flag = True
        if flag : print("STOP")
        self.loss_blur = self.criterionMSE(self.gamma_blur, self.blur_score)
        # print("loss_blur: ", self.loss_blur)
        # print("blur_score: ", self.blur_score)
        # print("gamma_blur: ", self.gamma_blur)

    def compute_losses(self):
        stylized_feats = self.encode_with_intermediate(self.cs)
        self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)
        self.compute_identity_loss()
        self.compute_blur_loss()
        self.loss_content = self.loss_content * self.opt.lambda_content
        # self.loss_local = self.loss_local * self.opt.lambda_local
        self.loss_global = self.loss_global * self.opt.lambda_global
        self.loss_blur = self.loss_blur * self.opt.lambda_blur
        self.loss_identity = self.loss_identity
        
    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_content + self.loss_identity + self.loss_global + self.loss_blur #+ self.loss_local
        # print("content: ", self.loss_content)
        # print("identity: ", self.loss_identity)
        # print("global: ", self.loss_global)
        # print("blur: ", self.loss_blur)
        loss.backward()
        self.optimizer_g.step()
