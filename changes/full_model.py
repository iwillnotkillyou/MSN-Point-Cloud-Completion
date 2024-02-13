from transfered_model import *
import emd.emd_module as emd


class FullModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, inputs, gt, eps, iters):
        output1, output2, expansion_penalty = self.model(inputs)
        gt = gt[:, :, :3]
        idx = np.random.randint(0, gt.shape[1], self.model.num_points)
        gt = gt[:, idx, :]
        dist, _ = self.EMD(output1, gt, eps, iters)
        emd1 = torch.sqrt(dist).mean(1)

        dist, _ = self.EMD(output2, gt, eps, iters)
        emd2 = torch.sqrt(dist).mean(1)

        return output1, output2, emd1, emd2, expansion_penalty
