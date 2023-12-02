import argparse

import numpy
import torch
import torch.onnx
from timm.utils.model import unwrap_model

from evolution import decode_cand_tuple, get_args_parser
from lib.config import cfg, update_config_from_file
from lib.datasets import build_dataset
from model.supernet_transformer import Vision_TransformerSuper

def main():
    info = torch.load('./result/evolution_0.9_20/checkpoint-6.pth.tar')

    top_accuracies_last_generation = info['top_accuracies'][-1]
    best_candidate_index = numpy.argmax(top_accuracies_last_generation)
    best_candidate = info['candidates'][best_candidate_index]

    depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(best_candidate)

    parser = argparse.ArgumentParser('AutoFormer evolution search', parents=[get_args_parser()])
    args = parser.parse_args()
    update_config_from_file(args.cfg)

    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args, folder_name="subImageNet")

    in_channels = 3

    model = Vision_TransformerSuper(img_size=args.input_size,
                                    patch_size=args.patch_size,
                                    in_chans=in_channels,
                                    embed_dim=cfg.SUPERNET.EMBED_DIM, depth=cfg.SUPERNET.DEPTH,
                                    num_heads=cfg.SUPERNET.NUM_HEADS,mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                                    qkv_bias=True, drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    gp=args.gp,
                                    num_classes=args.nb_classes,
                                    max_relative_position=args.max_relative_position,
                                    relative_position=args.relative_position,
                                    change_qkv=args.change_qkv, abs_pos=not args.no_abs_pos)

    sampled_config = {}
    sampled_config['layer_num'] = depth
    sampled_config['mlp_ratio'] = mlp_ratio
    sampled_config['num_heads'] = num_heads
    sampled_config['embed_dim'] = [embed_dim]*depth

    model_module = unwrap_model(model)
    model_module.set_sample_config(config=sampled_config)

    print(model)

    if False:
        model.eval()

        dummy_input = torch.zeros((1, in_channels, args.input_size, args.input_size))
        # Somehow this does not work
        # RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient
        torch.onnx.export(model, dummy_input, 'MCUFormer.onnx')

if __name__ == '__main__':
    main()
