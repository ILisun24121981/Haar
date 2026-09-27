import pyqtgraph as pg
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal


class CustomViewBox(pg.ViewBox):
    """Кастомный ViewBox из основного проекта для реализации сфокусированного зума."""

    def __init__(self, canvas_ref, parent=None):
        super().__init__(parent)
        self.canvas_ref = canvas_ref

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() in [Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier]:
            ev.ignore()
            return

        if self.canvas_ref.x_min_limit is None or self.canvas_ref.x_max_limit is None:
            ev.ignore()
            return

        mask = self.state['mouseEnabled']
        if not any(mask):
            ev.accept()
            return

        pos = self.mapSceneToView(ev.scenePos())
        delta = ev.delta()
        factor = 0.85 if delta > 0 else 1.15

        self.scaleBy(s=(factor, factor), center=pos)

        x_range, _ = self.viewRange()
        current_x_min, current_x_max = x_range
        current_width = current_x_max - current_x_min
        total_limit_width = self.canvas_ref.x_max_limit - self.canvas_ref.x_min_limit

        if current_width > total_limit_width:
            current_width = total_limit_width

        new_min = current_x_min
        new_max = current_x_min + current_width

        if new_min < self.canvas_ref.x_min_limit:
            new_min = self.canvas_ref.x_min_limit
            new_max = new_min + current_width
        if new_max > self.canvas_ref.x_max_limit:
            new_max = self.canvas_ref.x_max_limit
            new_min = new_max - current_width

        self.setXRange(new_min, new_max, padding=0)
        ev.accept()


class HaarAnalyzerView(QMainWindow):
    # Сигналы для связи с контроллером
    view_closed = pyqtSignal()
    marker_placed = pyqtSignal(int)
    period_changed = pyqtSignal()

    def __init__(self, file_name):
        super().__init__()
        self.setWindowTitle(f"Haar Analyzer Module - [{file_name}]")
        self.resize(1100, 650)

        self.x_min_limit = None
        self.x_max_limit = None
        self.raw_flow_data = None
        self.raw_indices = None
        self.markers = []

        # Шаг времени по умолчанию (20 мс)
        self.time_step_sec = 0.020

        self.init_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def init_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Битовый интервал (WEVLET_WINDOW_HALF):"))
        self.input_window = QLineEdit("10")
        self.input_window.setFixedWidth(60)
        control_layout.addWidget(self.input_window)

        control_layout.addWidget(QLabel("Период записи (SAMPLE_PERIOD_MS):"))
        self.input_period = QLineEdit("20")
        self.input_period.setFixedWidth(60)
        self.input_period.returnPressed.connect(self.period_changed.emit)
        control_layout.addWidget(self.input_period)

        self.btn_analyze = QPushButton("Анализ")
        self.btn_analyze.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px 15px;")
        control_layout.addWidget(self.btn_analyze)

        self.btn_reset_zoom = QPushButton("Сбросить зум")
        self.btn_reset_zoom.setStyleSheet("font-weight: bold; padding: 5px 15px;")
        control_layout.addWidget(self.btn_reset_zoom)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        custom_view_box = CustomViewBox(canvas_ref=self)
        self.plot_widget = pg.PlotWidget(viewBox=custom_view_box)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Расход (FlowRate)', color='#000000', size='11pt', bold=True)
        self.plot_widget.setLabel('bottom', 'Время (индексы отсчетов)', color='#000000', size='11pt', bold=True)
        main_layout.addWidget(self.plot_widget)

        self.plot_widget.scene().sigMouseClicked.connect(self.on_scene_clicked)

    def get_inputs(self):
        try:
            window = int(self.input_window.text())
            period = int(self.input_period.text())
            if window <= 0 or period <= 0:
                raise ValueError
            return window, period
        except ValueError:
            return None, None

    def fast_downsample(self, x_raw, y_raw, indices_raw, start_idx, end_idx, target_bins=4000):
        """Высокоточный MinMax-даунсемплинг из вашего проекта, сохраняющий оригинальные индексы."""
        n_points = end_idx - start_idx
        if n_points <= target_bins * 2 or target_bins < 2:
            return x_raw[start_idx:end_idx], y_raw[start_idx:end_idx], indices_raw[start_idx:end_idx]

        raw_bin_size = n_points // target_bins
        bin_size = max(2, (raw_bin_size // 2) * 2)

        actual_bins = n_points // bin_size
        if actual_bins <= 0:
            return x_raw[start_idx:end_idx], y_raw[start_idx:end_idx], indices_raw[start_idx:end_idx]

        has_tail = (n_points % bin_size) > 0
        total_bins = actual_bins + 1 if has_tail else actual_bins

        out_x = np.empty(total_bins * 2, dtype=np.float64)
        out_y = np.empty(total_bins * 2, dtype=np.float64)
        out_idx = np.empty(total_bins * 2, dtype=np.int64)

        for i in range(actual_bins):
            b_start = start_idx + i * bin_size
            b_end = b_start + bin_size
            y_sub = y_raw[b_start:b_end]

            local_min_idx = np.argmin(y_sub)
            local_max_idx = np.argmax(y_sub)
            min_idx = b_start + local_min_idx
            max_idx = b_start + local_max_idx

            if min_idx < max_idx:
                out_x[i * 2] = float(x_raw[min_idx])
                out_y[i * 2] = float(y_sub[local_min_idx])
                out_idx[i * 2] = indices_raw[min_idx]

                out_x[i * 2 + 1] = float(x_raw[max_idx])
                out_y[i * 2 + 1] = float(y_sub[local_max_idx])
                out_idx[i * 2 + 1] = indices_raw[max_idx]
            else:
                out_x[i * 2] = float(x_raw[max_idx])
                out_y[i * 2] = float(y_sub[local_max_idx])
                out_idx[i * 2] = indices_raw[max_idx]

                out_x[i * 2 + 1] = float(x_raw[min_idx])
                out_y[i * 2 + 1] = float(y_sub[local_min_idx])
                out_idx[i * 2 + 1] = indices_raw[min_idx]

        if has_tail:
            b_start = start_idx + actual_bins * bin_size
            b_end = end_idx
            y_sub = y_raw[b_start:b_end]
            local_min_idx = np.argmin(y_sub)
            local_max_idx = np.argmax(y_sub)
            min_idx = b_start + local_min_idx
            max_idx = b_start + local_max_idx
            idx_dest = actual_bins * 2
            if min_idx < max_idx:
                out_x[idx_dest] = float(x_raw[min_idx])
                out_y[idx_dest] = float(y_sub[local_min_idx])
                out_idx[idx_dest] = indices_raw[min_idx]

                out_x[idx_dest + 1] = float(x_raw[max_idx])
                out_y[idx_dest + 1] = float(y_sub[local_max_idx])
                out_idx[idx_dest + 1] = indices_raw[max_idx]
            else:
                out_x[idx_dest] = float(x_raw[max_idx])
                out_y[idx_dest] = float(y_sub[local_max_idx])
                out_idx[idx_dest] = indices_raw[max_idx]

                out_x[idx_dest + 1] = float(x_raw[min_idx])
                out_y[idx_dest + 1] = float(y_sub[local_min_idx])
                out_idx[idx_dest + 1] = indices_raw[min_idx]

        return out_x, out_y, out_idx

    def update_plot(self, flow_data):
        """Отрисовка кривой расхода с обязательным даунсемплингом по оси секунд."""
        self.clear_all_markers()
        self.plot_widget.clear()

        self.raw_y_visual = np.array(flow_data, dtype=np.float64)
        self.raw_indices = np.arange(len(flow_data), dtype=np.int64)

        # Пересчет индексов в физические секунды на лету
        self.raw_x_visual = self.raw_indices * self.time_step_sec

        if len(self.raw_x_visual) > 0:
            self.x_min_limit = float(np.min(self.raw_x_visual))
            self.x_max_limit = float(np.max(self.raw_x_visual))

            if self.x_min_limit == self.x_max_limit:
                self.x_max_limit += 1.0

            x_resampled, y_resampled, _ = self.fast_downsample(
                self.raw_x_visual, self.raw_y_visual, self.raw_indices,
                start_idx=0, end_idx=len(self.raw_x_visual), target_bins=4000
            )

            pen = pg.mkPen(color='#007acc', width=2)
            self.plot_widget.plot(x_resampled, y_resampled, pen=pen)
            self.reset_zoom()

    def draw_detected_bits(self, edges):
        """Отрисовка маркерных линий пачечного приема: START (зеленый), TRUE (красный), TECH (фиолетовый)."""
        for idx, state, marker_type in edges:
            time_pos = idx * self.time_step_sec
            y_pos = self.raw_y_visual[idx] if idx < len(self.raw_y_visual) else self.raw_y_visual[-1]

            # Формат вывода координаты X в зависимости от масштаба
            x_label_str = f"{time_pos:.1f}s" if self.time_step_sec != 1.0 else f"#{idx}"

            # Направление физического изменения расхода
            direction_str = "UP" if state else "DOWN"

            if marker_type == "START":
                # Первая зафиксированная точка синхронизации посылки (Зеленый с указанием UP/DOWN)
                line_pen = pg.mkPen(color='#27ae60', width=2, style=Qt.PenStyle.SolidLine)
                line = pg.InfiniteLine(pos=time_pos, angle=90, pen=line_pen, movable=False)
                self.plot_widget.addItem(line)

                bit_text = f"START ({direction_str})\n{x_label_str}"
                text_item = pg.TextItem(text=bit_text, color='#ffffff', anchor=(0.5, 1.3), fill='#27ae60')
                text_item.setPos(time_pos, y_pos)
                self.plot_widget.addItem(text_item)
                self.markers.append({'line': line, 'text': text_item, 'x': time_pos})

            elif marker_type == "TRUE":
                # Истинный бит Манчестера в середине интервала (Красный с указанием 1/0)
                line_pen = pg.mkPen(color='#e74c3c', width=1.5, style=Qt.PenStyle.DashLine)
                line = pg.InfiniteLine(pos=time_pos, angle=90, pen=line_pen, movable=False)
                self.plot_widget.addItem(line)

                bit_text = f"1 ({direction_str})\n{x_label_str}" if state else f"0 ({direction_str})\n{x_label_str}"
                text_item = pg.TextItem(text=bit_text, color='#ffffff', anchor=(0.5, 1.3), fill='#e74c3c')
                text_item.setPos(time_pos, y_pos)
                self.plot_widget.addItem(text_item)
                self.markers.append({'line': line, 'text': text_item, 'x': time_pos})

            elif marker_type == "TECH":
                # Технологический граничный фронт переключения насоса (Фиолетовый с указанием UP/DOWN)
                line_pen = pg.mkPen(color='#9b59b6', width=1.2, style=Qt.PenStyle.DotLine)
                line = pg.InfiniteLine(pos=time_pos, angle=90, pen=line_pen, movable=False)
                self.plot_widget.addItem(line)

                # Добавлено отображение направления для фиолетовых карточек TECH
                tech_text = f"TECH ({direction_str})\n{x_label_str}"
                text_item = pg.TextItem(text=tech_text, color='#ffffff', anchor=(0.5, 1.3), fill='#9b59b6')
                text_item.setPos(time_pos, y_pos)
                self.plot_widget.addItem(text_item)
                self.markers.append({'line': line, 'text': text_item, 'x': time_pos})

    def on_scene_clicked(self, event):
        """Интеграция установки маркеров по двойному клику с автоматическим поиском по времени."""
        if event.double():
            scene_pos = event.scenePos()
            view_box = self.plot_widget.getViewBox()
            plot_pos = view_box.mapSceneToView(scene_pos)
            click_x = plot_pos.x()

            if self.raw_x_visual is None or len(self.raw_x_visual) == 0:
                return

            x_range, _ = view_box.viewRange()
            threshold = (x_range - x_range) * 0.005

            for m in self.markers:
                if abs(m['x'] - click_x) <= threshold:
                    self.plot_widget.removeItem(m['line'])
                    if m['text'] is not None:
                        self.plot_widget.removeItem(m['text'])
                    self.markers.remove(m)
                    event.accept()
                    return

            closest_idx = np.argmin(np.abs(self.raw_x_visual - click_x))
            exact_x = self.raw_x_visual[closest_idx]
            exact_y = self.raw_y_visual[closest_idx]
            original_table_row = self.raw_indices[closest_idx]

            line_pen = pg.mkPen(color='#2c3e50', width=1.5, style=Qt.PenStyle.DashLine)
            line = pg.InfiniteLine(pos=exact_x, angle=90, pen=line_pen, movable=False)
            self.plot_widget.addItem(line)

            x_str = f"{exact_x:.1f}c" if self.time_step_sec != 1.0 else f"#{original_table_row}"
            label_text = f"T: {x_str}\nY: {exact_y:.4f}"
            text_item = pg.TextItem(text=label_text, color='#ffffff', anchor=(0, 1), fill='#2c3e50')
            text_item.setPos(exact_x, exact_y)
            self.plot_widget.addItem(text_item)

            self.markers.append({'line': line, 'text': text_item, 'x': exact_x})
            self.marker_placed.emit(int(original_table_row))
            event.accept()

    def wheelEvent(self, event):
        """Обработка модификаторов Ctrl (Прокрутка) и Shift (Горизонтальный Зум) по оси времени."""
        if self.x_min_limit is None or self.x_max_limit is None:
            event.ignore()
            return
        modifiers = event.modifiers()
        view_box = self.plot_widget.getViewBox()
        x_range, _ = view_box.viewRange()
        current_x_min, current_x_max = x_range
        current_width = current_x_max - current_x_min
        delta = event.angleDelta().y()
        if delta == 0:
            return

        step_factor = 0.1
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            shift_val = -current_width * step_factor if delta > 0 else current_width * step_factor
            new_min = current_x_min + shift_val
            new_max = current_x_max + shift_val
            if new_min < self.x_min_limit:
                new_min = self.x_min_limit
                new_max = new_min + current_width
            if new_max > self.x_max_limit:
                new_max = self.x_max_limit
                new_min = new_max - current_width
            view_box.setXRange(new_min, new_max, padding=0)
            event.accept()
        elif modifiers == Qt.KeyboardModifier.ShiftModifier:
            center = (current_x_min + current_x_max) / 2.0
            new_width = current_width * (1.0 + step_factor) if delta > 0 else current_width * (1.0 - step_factor)
            total_limit_width = self.x_max_limit - self.x_min_limit
            if new_width > total_limit_width:
                new_width = total_limit_width
            new_min = center - new_width / 2.0
            new_max = center + new_width / 2.0
            if new_min < self.x_min_limit:
                new_min = self.x_min_limit
                new_max = new_min + new_width
            if new_max > self.x_max_limit:
                new_max = self.x_max_limit
                new_min = new_max - new_width
            view_box.setXRange(new_min, new_max, padding=0)
            event.accept()
        else:
            super().wheelEvent(event)

    def reset_zoom(self):
        if self.raw_x_visual is not None and len(self.raw_x_visual) > 0:
            view_box = self.plot_widget.getViewBox()
            view_box.setRange(
                xRange=(self.x_min_limit, self.x_max_limit),
                yRange=(float(np.min(self.raw_y_visual)), float(np.max(self.raw_y_visual))),
                padding=0.02
            )

    def clear_all_markers(self):
        for m in self.markers:
            self.plot_widget.removeItem(m['line'])
            if m['text'] is not None:
                self.plot_widget.removeItem(m['text'])
        self.markers.clear()

    def closeEvent(self, event):
        self.view_closed.emit()
        event.accept()

    def cleanup(self):
        self.clear_all_markers()
        self.plot_widget.clear()
        self.plot_widget.setParent(None)
        self.plot_widget.deleteLater()
        self.main_widget.setParent(None)
        self.main_widget.deleteLater()
