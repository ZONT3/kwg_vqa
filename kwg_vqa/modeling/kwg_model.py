from torch import nn

import kwg_vqa.util.modeling as zm
from .kwg_text import KWGText
from .kwg_vision import KWGVision


class KWGModel(zm.Module):
    """
    Итоговая модель VQA
    Здесь все модули собираются воедино
    и определена архитектура "головы" модели.
    """

    def __init__(self, args):
        super().__init__(args)

        self.vision = KWGVision(args)
        self.text = KWGText(args)
        self.xmodal = KWGXModal(args)
        self.classifier = nn.Linear(args.hidden_size, args.num_labels)  # TODO num_labels

    def forward(self, batch):
        visual_data, text_data = batch
        # Для избежания обучения модуля зрения, используем уже закодированные изображения в датасете
        # visual_data = self.vision(visual_data)
        text_data = self.text(text_data)

        output = self.xmodal(visual_data, text_data)

        logits = self.classifier(output)
        loss = None  # TODO calculate loss

        return loss, logits


class KWGXModal(zm.Module):
    """
    Модуль кросс-модальности.
    Оперирует текстовыми и визуальными данными,
    возвращая их объединение.
    """

    def __init__(self, args):
        super().__init__(args)

    def forward(self, visual_data, text_data):
        pass
