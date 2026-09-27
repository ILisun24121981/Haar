from PyQt6.QtCore import QObject, pyqtSignal


class HaarAnalyzerController(QObject):
    # Прокси-сигнал для проброса индекса строки в главное приложение
    marker_synchronized = pyqtSignal(int)

    def __init__(self, model, view, on_close_callback=None):
        super().__init__()
        self.model = model
        self.view = view
        self.on_close_callback = on_close_callback

        # Привязка UI событий
        self.view.btn_analyze.clicked.connect(self.handle_analysis)
        self.view.btn_reset_zoom.clicked.connect(self.view.reset_zoom)
        self.view.view_closed.connect(self.handle_view_closed)

        # НОВОЕ: Обработчик нажатия Enter в поле ввода периода записи
        self.view.period_changed.connect(self.handle_period_changed)

        # Перенаправление клика по маркеру в основной симулятор
        self.view.marker_placed.connect(self.marker_synchronized.emit)

        # Вывод базового даунсемпл-графика
        self.view.update_plot(self.model.flow_data)

    def handle_period_changed(self):
        """Пересчитывает ось X в физические секунды и меняет подпись."""
        window_half, sample_period = self.view.get_inputs()
        if sample_period is None:
            self.view.show_error("Ошибка", "Введите корректный период записи в миллисекундах!")
            return

        # Переводим миллисекунды в секунды для шага сетки
        self.view.time_step_sec = sample_period / 1000.0

        # Меняем подпись оси на требуемую
        self.view.plot_widget.setLabel('bottom', 'Время (секунды)', color='#000000', size='11pt', bold=True)

        # Обновляем график в новом временном масштабе
        self.view.update_plot(self.model.flow_data)

    def handle_analysis(self):
        window_half, sample_period = self.view.get_inputs()
        if window_half is None or sample_period is None:
            self.view.show_error("Ошибка параметров", "Введите корректные целые положительные числа!")
            return

        # Гарантируем актуальный временной шаг перед запуском анализа
        self.view.time_step_sec = sample_period / 1000.0
        self.view.plot_widget.setLabel('bottom', 'Время (секунды)', color='#000000', size='11pt', bold=True)

        self.view.update_plot(self.model.flow_data)
        edges = self.model.run_haar_analysis(window_half, sample_period)

        if edges:
            self.view.draw_detected_bits(edges)
        else:
            self.view.show_info("Результат", "Битовые переходы Манчестера не обнаружены.")

    def handle_view_closed(self):
        """Полная выгрузка ссылок и очистка тяжелых объектов из RAM"""
        try:
            self.view.btn_analyze.clicked.disconnect()
            self.view.btn_reset_zoom.clicked.disconnect()
            self.view.view_closed.disconnect()
            self.view.marker_placed.disconnect()
            self.view.period_changed.disconnect()
        except TypeError:
            pass

        self.model.clear()
        self.view.cleanup()

        self.model = None
        self.view = None

        if self.on_close_callback:
            self.on_close_callback()
