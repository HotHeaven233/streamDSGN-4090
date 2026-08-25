import argparse
from pathlib import Path
import numpy as np
import torch
import json
import time

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--cfg_file',
        type=str,
        default='./configs/stereo/kitti_models/stream_dsgn_r18-sample.yaml',
        help='specify the config for training')
    parser.add_argument(
        '--exp_name',
        type=str,
        default='default',
        help='exp path for this experiment')
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='checkpoint to evaluate')
    parser.add_argument(
        '--ckpt_id',
        type=int,
        default=None,
        help='checkpoint id to evaluate')
    parser.add_argument(
        '--time_exist',
        action='store_true', 
        default=False, 
        help='')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '_'.join(args.cfg_file.split('/')[1:-1])
    cfg.DATA_CONFIG.INFER_TIME_PATH = str(Path('outputs') / 'inference_time' / cfg.EXP_GROUP_PATH / (cfg.TAG + '.json'))
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    # dataset
    test_set, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                      batch_size=1, dist=False, workers=4, logger=logger, training=False)
    logger.info(f'Total number of samples: \t{len(test_set)}')

    # model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=test_set)
    logger.info(model)

    # load ckpt
    if args.ckpt_id:
        assert args.exp_name
        output_dir = cfg.ROOT_DIR / 'outputs' / \
            cfg.EXP_GROUP_PATH / (cfg.TAG + '.' + args.exp_name)
        args.ckpt = str(output_dir / 'ckpt' /
                        'checkpoint_epoch_{}.pth'.format(args.ckpt_id))
    if args.ckpt is not None:
        model.load_params_from_file(
            filename=args.ckpt, logger=logger, to_cpu=True)
    
    time_files = Path(cfg.DATA_CONFIG.INFER_TIME_PATH)
    if args.time_exist:
        if not time_files.exists():
            logger.info('time file is not exist!')
        else:
            with open(time_files, 'r') as f:
                time_samples = json.load(f)
            time_samples = np.array([float(x) for x in time_samples])
            logger.info('mean model time: {}'.format(time_samples.mean()))

    else:
        model.cuda()
        model.eval()
        spend_time = []
        with torch.no_grad():
            for idx, data_dict in enumerate(test_set):
                data_dict = test_set.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                torch.cuda.synchronize()
                t1 = time.time()
                _, _ = model.forward(data_dict)
                torch.cuda.synchronize()
                t2 = time.time()
                spend_time.append(f'{(t2-t1)*1000:.3f}')
                logger.info(
                    f'Visualized sample index: \t{idx + 1}, \tspend time: \t{(t2-t1)*1000:.3f}')

                if idx >= 501:
                    break

        time_samples = np.array([float(x) for x in spend_time[1:]])
        logger.info('mean model time: {}'.format(time_samples.mean()))
        save_dir = time_files.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(time_files, 'w') as f:
            json.dump(spend_time[1:], f)


if __name__ == '__main__':
    main()
