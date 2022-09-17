from kwg_vqa.modeling.kwg_model import KWGModel

from .kwg_dataset import Dataset


class VQA:
    """
    Класс для работы с моделью KWGModel и датасетом.
    Здесь определены методы обучения и обработки результатов
    """

    def __init__(self, args):
        self.args = args
        self.dataset = Dataset(args)
        # self.model = KWGModel(args)

    def train(self):
        pass

    def val(self):
        pass

    def test(self):
        pass
