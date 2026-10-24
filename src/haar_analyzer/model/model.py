import numpy as np


class HaarDecoderModel:
    def __init__(self, flow_data):
        self.flow_data = np.array(flow_data, dtype=float)
        self.detected_edges = []

    def run_haar_analysis(self, window_half, sample_period):
        """
        РЕАЛТАЙМ-декодер гидравлического Манчестера.
        ВСЕ фронты фиксируются по критерию выхода в лимит отклонения от экстремума.
        Метка времени сдвигается ретроспективно ровно на СЕРЕДИНУ коридора удержания.
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

        # --- СКВОЗНЫЕ ПЕРЕМЕННЫЕ КРИТЕРИЯ ЛИМИТА ОТКЛОНЕНИЯ ---
        peak_D = 0.0  # Зафиксированный пиковый экстремум
        first_stable_tick_offset = 0  # Общее время нахождения внутри лимита (в тиках)
        in_stable_corridor = False  # Флаг удержания сигнала на полке вершины
        corridor_ticks_count = 0  # Счетчик времени нахождения в коридоре стабильности

        # Флаг того, что сигнал пробил порог амплитуды и движется к пику
        is_searching_peak = False
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

            ticks_since_any_edge += 1
            ticks_in_current_state += 1

            detected_edge = False
            next_state = current_state

            # --- СКВОЗНОЙ КРИТЕРИЙ ПИК-ДЕТЕКТОРА С ПРИВЯЗКОЙ К СЕРЕДИНЕ КОРИДОРА ---
            if ticks_in_current_state > deadtime_ticks:
                if not expect_down:
                    # 1. Ожидаем ПОДЪЕМ расхода
                    if not is_searching_peak:
                        if D > wave_threshold and d_derivative > 0.01:
                            is_searching_peak = True
                            peak_D = D
                    else:
                        if d_derivative > 0.01:
                            if D > peak_D:
                                peak_D = D
                            in_stable_corridor = False
                            corridor_ticks_count = 0
                        else:
                            # Проверяем лимит отклонения 15% от замершего пика
                            deviation_limit = peak_D * 0.15
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

                            # Критерий удержания вершины полки сигнала (около 2 секунд)
                            if corridor_ticks_count >= max(4, window_half // 8):
                                next_state = True
                                detected_edge = True
                                expect_down = True

                                # ИЗМЕНЕНО: Сдвигаем маркер назад строго на СЕРЕДИНУ коридора стабилизации
                                offset_to_middle = first_stable_tick_offset // 2
                                ticks_since_any_edge -= offset_to_middle
                                actual_edge_index = max(0, i - window_half - offset_to_middle)

                                # Полный сброс флагов для следующего такта
                                peak_D = 0.0
                                is_searching_peak = False
                                in_stable_corridor = False
                                corridor_ticks_count = 0
                else:
                    # 2. Ожидаем СПАД расхода
                    if not is_searching_peak:
                        if D < -wave_threshold and d_derivative < -0.01:
                            is_searching_peak = True
                            peak_D = D
                    else:
                        if d_derivative < -0.01:
                            if D < peak_D:
                                peak_D = D
                            in_stable_corridor = False
                            corridor_ticks_count = 0
                        else:
                            # Проверяем лимит отклонения от замершего минимума
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

                                # ИЗМЕНЕНО: Сдвигаем маркер спада назад строго на СЕРЕДИНУ коридора стабилизации
                                offset_to_middle = first_stable_tick_offset // 2
                                ticks_since_any_edge -= offset_to_middle
                                actual_edge_index = max(0, i - window_half - offset_to_middle)

                                peak_D = 0.0
                                is_searching_peak = False
                                in_stable_corridor = False
                                corridor_ticks_count = 0

            # Конечный автомат пачечной селекции Манчестера
            if detected_edge:
                dt = ticks_since_any_edge
                ticks_since_any_edge = 0
                ticks_in_current_state = 0
                current_state = next_state

                tolerance = int(HALF_PERIOD_TICKS * 0.35)
                marker_type = "TECH"

                if not is_synchronized:
                    # Стартовым может быть только фронт вниз (DOWN)
                    if current_state == False:
                        is_synchronized = True
                        true_bits_in_packet = 1
                        marker_type = "START"
                        self.detected_edges.append((actual_edge_index, current_state, marker_type))
                    else:
                        ticks_since_any_edge = 0
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
                            last_type = last_item[2]  # Извлекаем marker_type из кортежа истории
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
                        # --- НАРУШЕНИЕ КРАТНОСТИ / ОКОНЧАНИЕ ПОСЫЛКИ ---
                        if current_state == False:
                            if true_bits_in_packet < 3:
                                while self.detected_edges:
                                    popped = self.detected_edges.pop()
                                    if popped[2] == "START":
                                        break

                            true_bits_in_packet = 1
                            marker_type = "START"
                            is_synchronized = True
                            self.detected_edges.append((actual_edge_index, current_state, marker_type))
                        else:
                            is_synchronized = False
                            if true_bits_in_packet < 3:
                                while self.detected_edges:
                                    popped = self.detected_edges.pop()
                                    if popped[2] == "START":
                                        break
                            true_bits_in_packet = 0

                deadtime_ticks = max(4, int((BIT_PERIOD_TICKS / 2) * 0.12))

        return self.detected_edges

    def clear(self):
        """Полная выгрузка массивов из RAM"""
        self.flow_data = None
        self.detected_edges.clear()
