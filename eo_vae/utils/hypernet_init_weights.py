import os

import torch
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.models.dynamic_conv import (
    DynamicConv,
    DynamicConv_decoder,
    get_1d_sincos_pos_embed_from_grid_torch,
)

# DynamicConv(wv_planes=128, inter_dim=128, kernel_size=3, stride=1, padding=1, embed_dim=128)


class Dynamic_WG(DynamicConv):
    def __init__(
        self,
        wv_planes,
        inter_dim=128,
        kernel_size=3,
        stride=1,
        padding=1,
        embed_dim=128,
    ):
        super().__init__(wv_planes, inter_dim, kernel_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, waves):
        inplanes = waves.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        self.scaler = 0.01
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, waves * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])
        if bias is not None:
            bias = bias.view([self.embed_dim])

        weights = dynamic_weight * self.scaler
        bias = bias * self.scaler

        return weights, bias


class Dynamic_decoder(DynamicConv_decoder):
    def __init__(
        self,
        wv_planes,
        inter_dim=128,
        kernel_size=3,
        stride=1,
        padding=1,
        embed_dim=128,
    ):
        super().__init__(wv_planes, inter_dim, kernel_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, waves):
        inplanes = waves.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        self.scaler = 0.1
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, waves * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        # small bug fixed
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([0, 3, 1, 2])

        if bias is not None:
            bias = bias.squeeze() * self.scaler

        weights = dynamic_weight * self.scaler
        bias = bias * self.scaler

        return weights, bias


def train_one_epoch(epoch, model, t_model, waves, optimizer, criterion, scaler):
    # with torch.cuda.amp.autocast():
    # pdb.set_trace()
    weights, bias = model(waves.float())
    optimizer.zero_grad()
    target1 = t_model.weight.detach().clone().float()
    target2 = t_model.bias.detach().clone().float()

    loss = criterion(weights.float(), target1) + criterion(bias.float(), target2)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if epoch % 500 == 0:
        print('==============================================')
        print(target2[:8])
        print('---------------------------------------------')
        print(bias[:8])
        print('==============================================')
        print(f'Train Epoch: {epoch}, Loss: {loss.item():.6f}')


def main(epochs):
    config_path = os.path.join(os.getcwd(), 'configs', 'test_config.yaml')
    config = OmegaConf.load(config_path)
    t_model = instantiate(config.model).cuda()
    wavelist = [0.665, 0.560, 0.490]
    waves = torch.tensor(wavelist).cuda().view([3, 1])
    # waves = torch.tensor(wavelist).cuda()
    # weight_generator = Dynamic_WG(wv_planes=128, inter_dim=128, kernel_size=3, stride=1, padding=1, embed_dim=128).cuda()
    weight_generator = Dynamic_decoder(
        wv_planes=128, inter_dim=128, kernel_size=3, stride=1, padding=1, embed_dim=128
    ).cuda()

    optimizer = optim.AdamW(
        weight_generator.parameters(), lr=0.00001, weight_decay=0.0003
    )
    # criterion = nn.MSELoss()
    criterion = torch.nn.HuberLoss()
    scaler = torch.cuda.amp.GradScaler()

    # wg_weights = torch.load('weight_generator_init_0.01_er50k.pt')
    # weight_generator.weight_generator.load_state_dict(wg_weights["weight_generator"])
    # weight_generator.fclayer.load_state_dict(wg_weights["fclayer"])

    for i in range(epochs):
        # train_one_epoch(i, weight_generator, t_model.encoder.conv_in, waves, optimizer, criterion, scaler)
        train_one_epoch(
            i,
            weight_generator,
            t_model.decoder.conv_out,
            waves,
            optimizer,
            criterion,
            scaler,
        )

    state_dict = {
        'weight_generator': weight_generator.weight_generator.state_dict(),
        'fclayer': weight_generator.fclayer.state_dict(),
    }
    torch.save(state_dict, 'decoder_dconv_weight_generator_init_0.01_er50k.pt')


if __name__ == '__main__':
    epochs = 10000
    main(epochs)
