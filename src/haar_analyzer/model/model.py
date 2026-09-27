import numpy as np


class HaarDecoderModel:
    def __init__(self, flow_data):
        self.flow_data = np.array(flow_data, dtype=float)
        self.detected_edges = []

    def run_haar_analysis(self, window_half, sample_period):
        """
        Универсальный реалтайм-декодер гидравлического Манчестера.
        ВСЕ фронты (START/TRUE/TECH) фиксируются по критерию
        выхода в лимит отклонения от экстремума вейвлет-коэффициента.
        """
        total_samples = len(self.flow_data)
        self.detected_edges = []

        if total_samples == 0:
            return []

        # Потоковый кольцевой буфер истории вейвлета в ОЗУ
        buffer = [0.0] * (window_half * 2)
        head = 0
        left_sum = 0.0
        right_sum = 0.0

        # Переменные конечного автомата реалтайм-декодера
        ticks_since_any_edge = 0
        ticks_in_current_state = 0
        is_synchronized = False
        current_state = False

        # Константы битового интервала (36 секунд = 288 тиков при 125мс)
        BIT_PERIOD_TICKS = window_half * 2
        HALF_PERIOD_TICKS = window_half

        # Минимальное «мертвое время» защиты от дребезга
        deadtime_ticks = max(4, int(HALF_PERIOD_TICKS * 0.12))

        # Переменные следящего адаптивного порога
        local_amp = 0.0
        ALPHA_AMP = 0.05
        expect_down = False
        prev_D = 0.0
        trend_duration = 0

        # --- СКВОЗНЫЕ ПЕРЕМЕННЫЕ КРИТЕРИЯ ЛИМИТА ОТКЛОНЕНИЯ ДЛЯ ВСЕХ ФРОНТОВ ---
        peak_D = 0.0  # Пиковый экстремум текущей полуволны
        first_stable_tick_offset = 0  # Ретроспективный сдвиг к моменту завершения перехода
        in_stable_corridor = False  # Флаг удержания сигнала на полке вершины
        corridor_ticks_count = 0  # Счетчик времени нахождения в коридоре стабильности
        true_bits_in_packet = 0  # Счетчик истинных бит в текущей посылке

        # Потоковый прогон массива через Си-логику скользящего окна Хаара
        for i in range(total_samples):
            raw_flow = self.flow_data[i]

            # Пересчет сумм половин окна за O(1)
            tail_left = (head + 0) % (window_half * 2)
            mid_point = (head + window_half) % (window_half * 2)

            left_sum -= buffer[tail_left]
            left_sum += buffer[mid_point]

            right_sum -= buffer[mid_point]
            right_sum += raw_flow

            buffer[head] = raw_flow
            head = (head + 1) % (window_half * 2)

            # Вычисление коэффициента D и его мгновенной производной
            D = (right_sum - left_sum) / window_half
            d_derivative = D - prev_D
            prev_D = D

            # Потоковое слежение за локальной амплитудой для адаптации к затуханию
            local_amp = (ALPHA_AMP * abs(D)) + (1.0 - ALPHA_AMP) * local_amp
            wave_threshold = max(local_amp * 0.35, 1.5)

            # Валидатор тренда для защиты от ЧМ/ФМ-помех
            if not expect_down:
                if d_derivative > 0.01:
                    trend_duration += 1
                else:
                    trend_duration = 0
            else:
                if d_derivative < -0.01:
                    trend_duration += 1
                else:
                    trend_duration = 0

            ticks_since_any_edge += 1
            ticks_in_current_state += 1

            detected_edge = False
            next_state = current_state

            # --- СКВОЗНОЙ КРИТЕРИЙ ЛИМИТА ОТКЛОНЕНИЯ ДЛЯ ВСЕХ ФРОНТОВ ---
            if ticks_in_current_state > deadtime_ticks:
                if not expect_down:
                    # 1. Фиксация подъема: отслеживаем глобальный максимум волны
                    if D > wave_threshold and trend_duration >= max(2, window_half // 12):
                        if D > peak_D:
                            peak_D = D  # Обновляем пик нарастания

                        deviation_limit = peak_D * 0.15  # Лимит отклонения 15%

                        if abs(D - peak_D) <= deviation_limit:
                            if not in_stable_corridor:
                                in_stable_corridor = True
                                first_stable_tick_offset = 0
                            else:
                                first_stable_tick_offset += 1
                            corridor_ticks_count += 1
                        else:
                            in_stable_corridor = False
                            corridor_ticks_count = 0

                        # Критерий удержания: сигнал зафиксировался на полке
                        if corridor_ticks_count >= max(4, window_half // 8):
                            next_state = True
                            detected_edge = True
                            expect_down = True

                            # Корректируем интервал: отматываем счетчик назад к моменту завершения перехода
                            ticks_since_any_edge -= first_stable_tick_offset

                            # Сброс параметров слежения под следующий фронт
                            peak_D = 0.0
                            in_stable_corridor = False
                            corridor_ticks_count = 0
                            trend_duration = 0
                else:
                    # 2. Фиксация спада: отслеживаем глобальный минимум волны
                    if D < -wave_threshold and trend_duration >= max(2, window_half // 12):
                        if D < peak_D:
                            peak_D = D  # Обновляем пик спада

                        deviation_limit = abs(peak_D) * 0.15

                        if abs(D - peak_D) <= deviation_limit:
                            if not in_stable_corridor:
                                in_stable_corridor = True
                                first_stable_tick_offset = 0
                            else:
                                first_stable_tick_offset += 1
                            corridor_ticks_count += 1
                        else:
                            in_stable_corridor = False
                            corridor_ticks_count = 0

                        if corridor_ticks_count >= max(4, window_half // 8):
                            next_state = False
                            detected_edge = True
                            expect_down = False

                            ticks_since_any_edge -= first_stable_tick_offset

                            peak_D = 0.0
                            in_stable_corridor = False
                            corridor_ticks_count = 0
                            trend_duration = 0

            # Конечный автомат пачечной селекции Манчестера
            if detected_edge:
                dt = ticks_since_any_edge
                ticks_since_any_edge = 0
                ticks_in_current_state = 0
                current_state = next_state

                tolerance = int(HALF_PERIOD_TICKS * 0.35)
                actual_edge_index = max(0, i - window_half)
                marker_type = "TECH"

                if not is_synchronized:
                    is_synchronized = True
                    true_bits_in_packet = 1
                    marker_type = "START"
                    self.detected_edges.append((actual_edge_index, current_state, marker_type))
                else:
                    is_valid_interval = False
                    is_true_bit = False

                    # Проверка кратности интервалов (36с или 18с)
                    if abs(dt - BIT_PERIOD_TICKS) < tolerance:
                        is_valid_interval = True
                        is_true_bit = True
                    elif abs(dt - HALF_PERIOD_TICKS) < tolerance:
                        is_valid_interval = True
                        if self.detected_edges:
                            last_item = self.detected_edges[-1]
                            last_type = last_item[2]  # Извлекаем marker_type из кортежа
                            is_true_bit = (last_type == "TECH")
                        else:
                            is_true_bit = True

                    if is_valid_interval:
                        if is_true_bit:
                            true_bits_in_packet += 1
                            marker_type = "TRUE"
                        else:
                            marker_type = "TECH"
                        self.detected_edges.append((actual_edge_index, current_state, marker_type))
                    else:
                        # --- НАРУШЕНИЕ КРАТНОСТИ ---
                        if true_bits_in_packet >= 3:
                            true_bits_in_packet = 1
                            marker_type = "START"
                            self.detected_edges.append((actual_edge_index, current_state, marker_type))
                        else:
                            while self.detected_edges:
                                popped = self.detected_edges.pop()
                                if popped[2] == "START":
                                    break
                            true_bits_in_packet = 1
                            marker_type = "START"
                            self.detected_edges.append((actual_edge_index, current_state, marker_type))

                deadtime_ticks = max(4, int((BIT_PERIOD_TICKS / 2) * 0.12))

        return self.detected_edges

    def clear(self):
        """Полная выгрузка массивов из RAM"""
        self.flow_data = None
        self.detected_edges.clear()
