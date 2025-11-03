import torch
from .verifier import Verifier

verif_weights = torch.load('/home/n-lab/Amol/model_resnet18_100.pt')
self.verif_photo = Verifier().to(device)
self.verif_photo.load_state_dict(verif_weights['net_photo'])
self.verif_photo.eval()
for param in self.verif_photo.parameters():
    param.requires_grad = False
print('Verifier initialized...')

def compute_identity_loss(self):
    # self.loss_id = torch.tensor(0., device=self.device)
    # with torch.no_grad():
    real_embedding, real_features = self.verif_photo(self.c)
    stylized_embedding, stylized_features = self.verif_photo(self.cs)
    loss_verif_features = 0.0
    lambda_ridge_features = [1.0, 1.0, 1.0]
    for i in range(3):
        f_real = real_features[i]
        f_st = stylized_features[i]

        loss_verif_features += self.criterionMSE(f_st, f_real) * lambda_ridge_features[i]
    self.loss_id = self.criterionMSE(stylized_embedding, real_embedding) * 1.0 + loss_verif_features