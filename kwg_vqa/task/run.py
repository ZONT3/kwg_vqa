"""
Файл запуска проекта
"""

import kwg_vqa.task.arg as arg
from kwg_vqa.vqa.kwg_trainval import VQA

if __name__ == '__main__':
    args = arg.parse_args()
    vqa = VQA(args)

    if args.train:
        vqa.train()

    elif args.test:
        vqa.test()

    elif args.val:
        vqa.val()
