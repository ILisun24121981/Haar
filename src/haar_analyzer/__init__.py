from .model.model import HaarDecoderModel
from .view.view import HaarAnalyzerView
from .controller.controller import HaarAnalyzerController

def start_analyzer(flow_data, file_name, on_close_callback=None):
    """
    Запуск модуля анализа.
    Принимает массив данных, имя файла и callback-функцию, вызываемую при закрытии.
    """
    model = HaarDecoderModel(flow_data)
    view = HaarAnalyzerView(file_name)
    controller = HaarAnalyzerController(model, view, on_close_callback)
    view.show()
    return controller # Возвращаем контроллер, чтобы удерживать ссылки на MVC в RAM
