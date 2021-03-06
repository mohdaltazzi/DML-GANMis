import torch
from datetime import datetime
from pathlib import Path
import argparse
from mnist_generator import (ConvDataGenerator, FCDataGenerator,
                             ConvMaskGenerator, FCMaskGenerator)
from mnist_critic import ConvCritic, FCCritic
from masked_mnist import IndepMaskedMNIST, BlockMaskedMNIST
from misgan import misgan



use_cuda = torch.cuda.is_available()
use_cuda=0
device = torch.device('cuda' if use_cuda else 'cpu')


print(device)

def main():
    parser = argparse.ArgumentParser()

    # resume from checkpoint
    parser.add_argument('--resume')
    # training options
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)

    # log options: 0 to disable plot-interval or save-interval
    parser.add_argument('--plot-interval', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=0)
    parser.add_argument('--prefix', default='misgan')

    # mask options (data): block|indep
    parser.add_argument('--mask', default='block')
    # option for block: set to 0 for variable size
    parser.add_argument('--block-len', type=int, default=14)
    # option for indep:
    parser.add_argument('--obs-prob', type=float, default=.2)
    parser.add_argument('--obs-prob-high', type=float, default=None)

    # model options
    parser.add_argument('--tau', type=float, default=0)
    parser.add_argument('--generator', default='conv')   # conv|fc
    parser.add_argument('--critic', default='conv')   # conv|fc
    # parser.add_argument('--alpha', type=float, default=.1)   # 0: separate
    parser.add_argument('--alpha', type=float, default=.2)   # 0: separate
    # options for mask generator: sigmoid, hardsigmoid, fusion
    # parser.add_argument('--maskgen', default='fusion')
    parser.add_argument('--maskgen', default='sigmoid')
    parser.add_argument('--gp-lambda', type=float, default=10)
    parser.add_argument('--n-critic', type=int, default=5)
    parser.add_argument('--n-latent', type=int, default=128)

    args = parser.parse_args()

    checkpoint = None
    # Resume from previously stored checkpoint
    if args.resume:
        print(f'Resume: {args.resume}')
        output_dir = Path(args.resume)
        checkpoint = torch.load(str(output_dir / 'log' / 'checkpoint.pth'),
                                map_location='cpu')
        for key, arg in vars(checkpoint['args']).items():
            if key not in ['resume']:
                setattr(args, key, arg)

    if args.generator == 'conv':
        DataGenerator = ConvDataGenerator
        MaskGenerator = ConvMaskGenerator
    elif args.generator == 'fc':
        DataGenerator = FCDataGenerator
        MaskGenerator = FCMaskGenerator
    else:
        raise NotImplementedError

    if args.critic == 'conv':
        Critic = ConvCritic
    elif args.critic == 'fc':
        Critic = FCCritic
    else:
        raise NotImplementedError

    if args.maskgen == 'sigmoid':
        hard_sigmoid = False
    elif args.maskgen == 'hardsigmoid':
        hard_sigmoid = True
    elif args.maskgen == 'fusion':
        hard_sigmoid = -.1, 1.1
    else:
        raise NotImplementedError

    mask = args.mask
    obs_prob = args.obs_prob
    obs_prob_high = args.obs_prob_high
    block_len = args.block_len
    if block_len == 0:
        block_len = None

    if mask == 'indep':
        if obs_prob_high is None:
            mask_str = f'indep_{obs_prob:g}'
        else:
            mask_str = f'indep_{obs_prob:g}_{obs_prob_high:g}'
    elif mask == 'block':
        mask_str = 'block_{}'.format(block_len if block_len else 'varsize')
    else:
        raise NotImplementedError

    path = '{}_{}_{}'.format(
        args.prefix, datetime.now().strftime('%m%d.%H%M%S'),
        '_'.join([
            f'gen_{args.generator}',
            f'critic_{args.critic}',
            f'tau_{args.tau:g}',
            f'alpha_{args.alpha:g}',
            f'maskgen_{args.maskgen}',
            mask_str,
        ]))

    if not args.resume:
        output_dir = Path('results') / 'mnist' / path
        print(output_dir)

    if mask == 'indep':
        data = IndepMaskedMNIST(obs_prob=obs_prob, obs_prob_high=obs_prob_high)
    elif mask == 'block':
        data = BlockMaskedMNIST(block_len=block_len)

    data_gen = DataGenerator().to(device)
    mask_gen = MaskGenerator(hard_sigmoid=hard_sigmoid).to(device)

    data_critic = Critic().to(device)
    mask_critic = Critic().to(device)

    misgan(args, data_gen, mask_gen, data_critic, mask_critic, data,
           output_dir, checkpoint)


if __name__ == '__main__':
    main()
