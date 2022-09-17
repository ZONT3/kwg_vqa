"""
Парсинг аргументов
"""

from argparse import ArgumentParser

p = ArgumentParser()

p.add_argument('--train', action='store_true', help="Do train")
p.add_argument('--test', action='store_true', help="Do test")
p.add_argument('--val', action='store_true', help="Do val")

p.add_argument('--tiny', action='store_true', help="Tiny dataset size for experiments")
p.add_argument('--update-features', action='store_true', help="Extract and save features even if it was done before")
p.add_argument('--yes', '-y', action='store_true', help="Yes to all prompts")

p.add_argument('--dataset', type=str, required=True, help='Dataset directory. Must contain \'text\' subdirectory and '
                                                          'one of (or both) \'img\', \'feats\' subdirectories')

p.add_argument('--device', type=str, default=None, help='Select device explicitly. May be \'cuda\' or \'cpu\'. '
                                                        'Auto-detect by default')
p.add_argument('--batch-size', type=int, default=32, help='Batch size')
p.add_argument('--hidden-size', type=int, default=768, help='Hidden layer size')
p.add_argument('--vision-model-path', type=str, default='vinvl_vision', help='Path to download/read VinVL '
                                                                             'vision model weights and config')


def parse_args(src=None):
    return p.parse_args(src)
