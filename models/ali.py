import torch
import torch.nn as nn
import torch.nn.parallel

eps = 1e-8


def softplus(_x):
    return torch.log(1.0 + torch.exp(_x))


def compute_ali_losses(D, x, z_pred, z, x_pred):
    D_x, D_z, D_xz = D

    D_x_z_pred = D_xz(torch.cat([D_x(x), D_z(z_pred)], 1)) + eps
    D_x_pred_z = D_xz(torch.cat([D_x(x_pred), D_z(z)], 1)) + eps

    # discriminator loss
    loss_D = torch.mean(softplus(-D_x_z_pred) + softplus(D_x_pred_z))

    # generator loss
    loss_G = torch.mean(softplus(D_x_z_pred) + softplus(-D_x_pred_z))

    return loss_D, loss_G


def create_ali_D(nz=512, ngpu=1):
    D_x = create_dx(ngpu)
    D_z = create_dz(nz, ngpu)
    D_xz = create_dxz(ngpu)
    return nn.ModuleList([D_x, D_z, D_xz])


def create_dx(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // conv. bias // batch_norm // dropout // nonlinearity
        ['conv2d', 5, 1,  32, True,  False, 0.2, 'leaky_relu'],
        ['conv2d', 4, 2,  64, False, True, 0.2, 'leaky_relu'],
        ['conv2d', 4, 1, 128, False, True, 0.2, 'leaky_relu'],
        ['conv2d', 4, 2, 256, False, True, 0.2, 'leaky_relu'],
        ['conv2d', 4, 1, 512, False, True, 0.2, 'leaky_relu'],
    ]
    return CNN(3, 32, hparams, ngpu)


def create_dz(nz=512, ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // conv. bias // batch_norm // dropout // nonlinearity
        ['conv2d', 3, 2, 512, False, False, 0.2, 'leaky_relu'],
        ['conv2d', 3, 2, 512, False, False, 0.2, 'leaky_relu'],
    ]
    return CNN(nz, 8, hparams, ngpu)


def create_dxz(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // conv. bias // batch_norm // dropout // nonlinearity
        ['conv2d', 1, 1, 1024, True, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1, 1024, True, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1,    1, True, False, 0.2, 'linear'],
    ]
    return CNN(1024, 1, hparams, ngpu)


class CNN(nn.Module):
    def __init__(self, nc, input_size, hparams, ngpu=1, leaky_slope=0.01, std=0.01):
        super(CNN, self).__init__()
        self.ngpu = ngpu  # num of gpu's to use
        self.leaky_slope = leaky_slope  # slope for leaky_relu activation
        self.std = std  # standard deviation for weights initialization
        self.input_size = input_size  # expected input size

        main = nn.Sequential()
        in_feat, num = nc, 0
        for op, k, s, out_feat, b, bn, dp, h in hparams:
            # add operation: conv2d or convTranspose2d
            if op == 'conv2d':
                main.add_module(
                    '{0}_pyramid_{1}-{2}_conv'.format(num, in_feat, out_feat),
                    nn.Conv2d(in_feat, out_feat, k, s, 0, bias=b))
            elif op == 'convt2d':
                main.add_module(
                    '{0}_pyramid_{1}-{2}_convt'.format(num,in_feat, out_feat),
                    nn.ConvTranspose2d(in_feat, out_feat, k, s, 0, bias=b))
            else:
                raise Exception('Not supported operation: {0}'.format(op))
            num += 1
            # add batch normalization layer
            if bn:
                main.add_module(
                    '{0}_pyramid_{1}-{2}_batchnorm'.format(num, in_feat, out_feat),
                    nn.BatchNorm2d(out_feat))
                num += 1
            # add dropout layer
            main.add_module(
                '{0}_pyramid_{1}-{2}_dropout'.format(num, in_feat, out_feat),
                nn.Dropout2d(p=dp))
            num += 1
            # add activation
            if h == 'leaky_relu':
                main.add_module(
                    '{0}_pyramid_{1}-{2}_leaky_relu'.format(num, in_feat, out_feat),
                    nn.LeakyReLU(self.leaky_slope, inplace=True))
            elif h == 'sigmoid':
                main.add_module(
                    '{0}_pyramid_{1}-{2}_sigmoid'.format(num, in_feat, out_feat),
                    nn.Sigmoid())
            elif h == 'maxout':
                # TODO: implement me
                # https://github.com/IshmaelBelghazi/ALI/blob/master/ali/bricks.py#L338-L380
                raise NotImplementedError('Maxout is not implemented.')
            elif h == 'relu':
                main.add_module(
                    '{0}_pyramid_{1}-{2}_relu'.format(num, in_feat, out_feat),
                    nn.ReLU(inplace=True))
            elif h == 'tanh':
                main.add_module(
                    '{0}_pyramid_{1}-{2}_tanh'.format(num, in_feat, out_feat),
                    nn.Tanh())
            elif h == 'linear':
                num -= 1  # 'Linear' do nothing
            else:
                raise Exception('Not supported activation: {0}'.format(h))
            num += 1
            in_feat = out_feat
        self.main = main

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, self.std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, self.std)
                m.bias.data.zero_()

    def forward(self, input):
        assert input.size(2) == self.input_size,\
            'Wrong input size: {0}. Expected {1}'.format(input.size(2),
                                                         self.input_size)
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)
        return output
