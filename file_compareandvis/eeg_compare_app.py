#!/usr/bin/env python3
"""
EEG Compare & Visualization GUI Application
Сравнение и наложение двух файлов ЭЭГ (.fif и .set):
- Интерактивный сдвиг во времени с шагом 50 мс (и 200, 500, 1000 мс)
- Обрезание каждого графика с начала и с конца на заданное время
- Точный расчет корреляции Пирсона и RMSE по обрезанным кусочкам
- Поканальная визуализация и детальный отчет корреляции
- Выбор каналов в один клик, фильтрация, пагинация
- Отображение и навигация по меткам (annotations / triggers / events)
"""

import sys
import os
import csv
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import mne

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def load_eeg_file(file_path: str) -> mne.io.BaseRaw:
    """Загружает ЭЭГ файл формата .fif или .set через MNE."""
    path_str = str(file_path)
    if not os.path.isfile(path_str):
        raise FileNotFoundError(f"Файл не найден: {path_str}")

    lower = path_str.lower()
    if lower.endswith(".set"):
        raw = mne.io.read_raw_eeglab(path_str, preload=True, verbose="ERROR")
    elif lower.endswith(".fif") or lower.endswith(".fif.gz"):
        raw = mne.io.read_raw_fif(path_str, preload=True, verbose="ERROR")
    else:
        raw = mne.io.read_raw(path_str, preload=True, verbose="ERROR")

    return raw


def extract_raw_markers(raw: mne.io.BaseRaw, file_path: Optional[str] = None) -> List[Tuple[float, float, str]]:
    """Извлекает метки событий (onset_sec, duration_sec, description)."""
    markers: List[Tuple[float, float, str]] = []
    sf = raw.info["sfreq"]

    if raw.annotations is not None and len(raw.annotations) > 0:
        for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            markers.append((float(onset), float(dur), str(desc).strip()))

    try:
        events = mne.find_events(raw, stim_channel=None, verbose="ERROR")
        if events is not None and len(events) > 0:
            for row in events:
                sample_idx, _, event_id = row
                onset = float(sample_idx / sf)
                if not any(abs(m[0] - onset) < 0.005 for m in markers):
                    markers.append((onset, 0.0, f"Trig_{event_id}"))
    except Exception:
        pass

    if not markers and file_path:
        p = Path(file_path)
        stem = p.stem.replace(".fif", "").replace("_clean", "").replace("_raw", "")
        candidates = [
            p.with_suffix(".tsv"),
            p.with_name(f"{stem}_events.tsv"),
            p.with_name("events.tsv"),
            p.with_suffix(".vmrk"),
            p.with_name(f"{stem}.vmrk"),
            p.with_suffix(".csv"),
            p.with_name(f"{stem}_events.csv"),
            p.parent / f"{stem}_events.tsv",
            p.parent / "events.tsv",
        ]
        for cand in candidates:
            if cand.is_file():
                loaded = parse_external_markers(str(cand), sfreq=sf)
                if loaded:
                    markers.extend(loaded)
                    break

    markers.sort(key=lambda x: x[0])
    return markers


def parse_external_markers(file_path: str, sfreq: float = 1000.0) -> List[Tuple[float, float, str]]:
    """Парсит внешний файл меток: .tsv, .csv, .vmrk или .txt."""
    p = Path(file_path)
    res: List[Tuple[float, float, str]] = []

    if p.suffix.lower() == ".vmrk":
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Mk") and "=" in line:
                    parts = line.split("=", 1)[1].split(",")
                    if len(parts) >= 3:
                        m_type = parts[0].strip()
                        m_desc = parts[1].strip() or m_type
                        try:
                            sample_pos = int(parts[2].strip())
                            dur_samples = int(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else 0
                            res.append((sample_pos / sfreq, dur_samples / sfreq, m_desc))
                        except ValueError:
                            pass
    else:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = "\t" if "\t" in sample else ("," if "," in sample else " ")
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader and reader.fieldnames:
                for row in reader:
                    onset = None
                    for k in ["onset", "latency", "time", "Time", "Onset", "Latency", "start", "position"]:
                        if k in row and row[k]:
                            try:
                                onset = float(row[k])
                                break
                            except ValueError:
                                pass
                    if onset is not None:
                        dur = 0.0
                        for k in ["duration", "Duration", "dur", "length", "Length"]:
                            if k in row and row[k]:
                                try:
                                    dur = float(row[k])
                                    break
                                except ValueError:
                                    pass
                        desc = (row.get("trial_type") or row.get("value") or
                                row.get("description") or row.get("type") or "Marker")
                        res.append((onset, dur, str(desc).strip()))

    res.sort(key=lambda x: x[0])
    return res


class MplCanvas(FigureCanvas):
    """Холст Matplotlib для отображения трасс ЭЭГ сигналов и меток."""
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("#1e1e24")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#18181b")
        super().__init__(self.fig)
        self.setParent(parent)
        self.figure.tight_layout()


class ChannelListWidget(QtWidgets.QListWidget):
    """Интерактивный список каналов с переключением кликом в любую часть строки."""
    def __init__(self, app_window: "EEGCompareApp", parent=None):
        super().__init__(parent)
        self.app_window = app_window
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        item = self.itemAt(event.pos())
        if item is not None and event.button() == Qt.LeftButton:
            rect = self.visualItemRect(item)
            if event.pos().x() > rect.left() + 24:
                new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                item.setCheckState(new_state)
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_Space:
            item = self.currentItem()
            if item is not None:
                new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                item.setCheckState(new_state)
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_item_double_clicked(self, item: QtWidgets.QListWidgetItem):
        if item is not None:
            self.app_window.solo_channel(item.text())

    def _show_context_menu(self, pos: QPoint):
        item = self.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #1e1e24; color: #e2e8f0; border: 1px solid #334155; padding: 4px; }
            QMenu::item { padding: 5px 15px; }
            QMenu::item:selected { background-color: #2563eb; color: white; }
        """)
        if item is not None:
            ch_name = item.text()
            act_solo = menu.addAction(f"Только этот канал: {ch_name} (Solo)")
            act_toggle = menu.addAction(f"Переключить {ch_name}")
            menu.addSeparator()
            act_all_vis = menu.addAction("Включить все видимые в фильтре")
            act_none_vis = menu.addAction("Отключить все видимые в фильтре")

            action = menu.exec_(self.mapToGlobal(pos))
            if action == act_solo:
                self.app_window.solo_channel(ch_name)
            elif action == act_toggle:
                new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                item.setCheckState(new_state)
            elif action == act_all_vis:
                self.app_window.set_visible_filter_channels_checked(True)
            elif action == act_none_vis:
                self.app_window.set_visible_filter_channels_checked(False)


class MarkersDialog(QtWidgets.QDialog):
    """Диалоговое окно полного списка меток с поиском и переходом по клику."""
    def __init__(self, app_window: "EEGCompareApp", parent=None):
        super().__init__(parent)
        self.app_window = app_window
        self.setWindowTitle("Все метки и события файлов")
        self.resize(550, 500)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e24; color: #e2e8f0; }
            QTableWidget { background-color: #0f172a; color: #f8fafc; gridline-color: #334155; border: 1px solid #334155; }
            QHeaderView::section { background-color: #1e293b; color: #93c5fd; padding: 4px; border: 1px solid #334155; font-weight: bold; }
            QLineEdit { background-color: #0f172a; border: 1px solid #475569; padding: 4px 8px; color: #f8fafc; }
            QPushButton { background-color: #2563eb; color: white; padding: 6px 14px; border-radius: 4px; font-weight: bold; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Фильтр по названию метки...")
        self.search_edit.textChanged.connect(self._filter_table)
        layout.addWidget(self.search_edit)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Файл", "Время (с)", "Длительность (с)", "Метка / Описание"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table)

        btn_row = QtWidgets.QHBoxLayout()
        btn_jump = QtWidgets.QPushButton("Перейти к выбранной метке")
        btn_jump.clicked.connect(self._jump_selected)
        btn_close = QtWidgets.QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_jump)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self._populate()

    def _populate(self):
        rows = []
        for m in self.app_window.markers1:
            rows.append(("Файл 1", m[0], m[1], m[2]))
        for m in self.app_window.markers2:
            rows.append(("Файл 2", m[0] + self.app_window.time_offset2, m[1], m[2]))

        rows.sort(key=lambda x: x[1])
        self.table.setRowCount(len(rows))
        for i, (f_name, onset, dur, desc) in enumerate(rows):
            it_file = QtWidgets.QTableWidgetItem(f_name)
            it_file.setForeground(QtGui.QBrush(QtGui.QColor("#60a5fa" if f_name == "Файл 1" else "#f59e0b")))
            it_time = QtWidgets.QTableWidgetItem(f"{onset:.3f}")
            it_dur = QtWidgets.QTableWidgetItem(f"{dur:.3f}")
            it_desc = QtWidgets.QTableWidgetItem(desc)

            self.table.setItem(i, 0, it_file)
            self.table.setItem(i, 1, it_time)
            self.table.setItem(i, 2, it_dur)
            self.table.setItem(i, 3, it_desc)

    def _filter_table(self, txt: str):
        txt = txt.lower().strip()
        for i in range(self.table.rowCount()):
            desc = self.table.item(i, 3).text().lower()
            file_lbl = self.table.item(i, 0).text().lower()
            self.table.setRowHidden(i, not (txt in desc or txt in file_lbl))

    def _on_row_double_clicked(self, item):
        self._jump_selected()

    def _jump_selected(self):
        cur_row = self.table.currentRow()
        if cur_row >= 0:
            time_str = self.table.item(cur_row, 1).text()
            try:
                t = float(time_str)
                target_t = max(0.0, t - self.app_window.win_duration / 2.0)
                self.app_window._set_time(target_t)
            except ValueError:
                pass


class CorrelationDialog(QtWidgets.QDialog):
    """Детальное диалоговое окно расчета корреляции по каналам и кусочкам."""
    def __init__(self, stats_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.stats = stats_data
        self.setWindowTitle("Анализ корреляции между графиками (по обрезанным кусочкам)")
        self.resize(850, 650)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e24; color: #e2e8f0; }
            QTableWidget { background-color: #0f172a; color: #f8fafc; gridline-color: #334155; border: 1px solid #334155; }
            QHeaderView::section { background-color: #1e293b; color: #93c5fd; padding: 4px; border: 1px solid #334155; font-weight: bold; }
            QPushButton { background-color: #2563eb; color: white; padding: 6px 14px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton.secondary { background-color: #334155; color: #f1f5f9; }
            QLabel { color: #cbd5e1; }
        """)

        layout = QtWidgets.QVBoxLayout(self)

        # Сводные карточки
        cards_layout = QtWidgets.QHBoxLayout()

        def make_card(title: str, val: str, color: str = "#60a5fa") -> QtWidgets.QFrame:
            frame = QtWidgets.QFrame()
            frame.setStyleSheet(f"background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 8px;")
            l = QtWidgets.QVBoxLayout(frame)
            lbl_t = QtWidgets.QLabel(title)
            lbl_t.setStyleSheet("font-size: 11px; color: #94a3b8;")
            lbl_v = QtWidgets.QLabel(val)
            lbl_v.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
            l.addWidget(lbl_t)
            l.addWidget(lbl_v)
            return frame

        mean_r = self.stats["mean_r"]
        mean_rmse = self.stats["mean_rmse"]
        r_color = "#10b981" if mean_r >= 0.8 else ("#f59e0b" if mean_r >= 0.5 else "#ef4444")

        cards_layout.addWidget(make_card("Средняя Корреляция (r)", f"{mean_r:.4f}", r_color))
        cards_layout.addWidget(make_card("Средняя RMSE", f"{mean_rmse:.2f} мкВ", "#60a5fa"))
        cards_layout.addWidget(make_card("Диапазон r", f"[{self.stats['min_r']:.3f} ... {self.stats['max_r']:.3f}]", "#a855f7"))
        cards_layout.addWidget(make_card("Длительность кусочка", f"{self.stats['eff_duration']:.2f} с ({self.stats['n_samples']} тчк)", "#38bdf8"))
        layout.addLayout(cards_layout)

        # Таблица каналов
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Канал", "Pearson r", "RMSE (мкВ)", "Оценка сходства"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(True)

        ch_data = self.stats["channel_stats"]
        self.table.setRowCount(len(ch_data))
        for i, row in enumerate(ch_data):
            ch_item = QtWidgets.QTableWidgetItem(row["channel"])

            # Для правильной сортировки чисел используем кастомные данные
            r_item = QtWidgets.QTableWidgetItem()
            r_item.setData(Qt.DisplayRole, round(row["r"], 4))
            r_color = QtGui.QColor("#10b981" if row["r"] >= 0.8 else ("#f59e0b" if row["r"] >= 0.5 else "#ef4444"))
            r_item.setForeground(QtGui.QBrush(r_color))

            rmse_item = QtWidgets.QTableWidgetItem()
            rmse_item.setData(Qt.DisplayRole, round(row["rmse"], 2))

            quality = "Высокое (r ≥ 0.8)" if row["r"] >= 0.8 else ("Умеренное (0.5 ≤ r < 0.8)" if row["r"] >= 0.5 else "Низкое / Расхождение")
            q_item = QtWidgets.QTableWidgetItem(quality)
            q_item.setForeground(QtGui.QBrush(r_color))

            self.table.setItem(i, 0, ch_item)
            self.table.setItem(i, 1, r_item)
            self.table.setItem(i, 2, rmse_item)
            self.table.setItem(i, 3, q_item)

        layout.addWidget(self.table)

        # Нижняя панель с кнопками
        btn_layout = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Экспорт таблицы в CSV...")
        btn_export.setProperty("class", "secondary")
        btn_export.clicked.connect(self._export_csv)

        btn_close = QtWidgets.QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(btn_export)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def _export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить отчет корреляции", "eeg_correlation_report.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Channel", "Pearson_r", "RMSE_uV"])
                for row in self.stats["channel_stats"]:
                    writer.writerow([row["channel"], f"{row['r']:.6f}", f"{row['rmse']:.4f}"])
            QtWidgets.QMessageBox.information(self, "Успешно", f"Отчет сохранен в {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка сохранения", str(e))


class EEGCompareApp(QtWidgets.QMainWindow):
    """Главное окно приложения сравнения ЭЭГ."""

    def __init__(self, file1: Optional[str] = None, file2: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("EEG Compare & Visualizer (.fif / .set) — With Trimming & Correlation")
        self.resize(1420, 920)

        # Данные ЭЭГ
        self.raw1: Optional[mne.io.BaseRaw] = None
        self.raw2: Optional[mne.io.BaseRaw] = None
        self.file1_path: Optional[str] = None
        self.file2_path: Optional[str] = None

        # Метки / аннотации
        self.markers1: List[Tuple[float, float, str]] = []
        self.markers2: List[Tuple[float, float, str]] = []
        self.show_markers: bool = True
        self.show_markers1: bool = True
        self.show_markers2: bool = True
        self.marker_filter: str = "Все типы"

        # Сдвиг во времени (Time shift)
        self.time_offset2: float = 0.0      # Сдвиг файла 2 относительно файла 1 (сек)
        self.shift_step_ms: float = 50.0    # Шаг сдвига по умолчанию 50 мс

        # Обрезание графиков (Trimming)
        self.trim_enabled: bool = False
        self.trim_f1_start: float = 0.0
        self.trim_f1_end: float = 0.0
        self.trim_f2_start: float = 0.0
        self.trim_f2_end: float = 0.0

        # Параметры отображения
        self.time_pos: float = 0.0          # Текущее время окна (сек)
        self.win_duration: float = 5.0      # Длительность окна (сек)
        self.scale_uv: float = 50.0         # Масштаб в мкВ (размах на канал)
        self.display_mode: str = "overlay"  # "overlay", "diff", "split"
        self.channels_per_page: int = 16    # Каналов на странице
        self.current_page: int = 0
        self.selected_channels: List[str] = []

        # Построение интерфейса
        self._init_theme()
        self._init_ui()

        # Загрузка переданных файлов при старте
        if file1 and os.path.exists(file1):
            self.load_file(1, file1)
        if file2 and os.path.exists(file2):
            self.load_file(2, file2)

    def _init_theme(self):
        """Темная стильная тема оформления."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e24;
                color: #e2e8f0;
                font-family: 'Segoe UI', 'Ubuntu', 'Helvetica', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #93c5fd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e40af; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
            QPushButton.secondary { background-color: #334155; color: #f1f5f9; }
            QPushButton.secondary:hover { background-color: #475569; }
            QPushButton.accent { background-color: #059669; color: white; }
            QPushButton.accent:hover { background-color: #047857; }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
                background-color: #0f172a;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
                color: #f8fafc;
            }
            QListWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 4px;
                color: #f8fafc;
                outline: none;
            }
            QListWidget::item { padding: 4px 6px; border-radius: 3px; margin: 1px 2px; }
            QListWidget::item:hover { background-color: #1e293b; }
            QListWidget::item:selected { background-color: #3b82f6; color: white; }
            QCheckBox { spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QSlider::groove:horizontal { height: 6px; background: #334155; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #2563eb; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #60a5fa; border: 1px solid #2563eb;
                width: 16px; margin-top: -5px; margin-bottom: -5px; border-radius: 8px;
            }
            QStatusBar { background-color: #0f172a; color: #94a3b8; border-top: 1px solid #334155; }
            QLabel { color: #cbd5e1; }
        """)

    def _init_ui(self):
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Левая панель с прокруткой QScrollArea
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(375)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)

        # 1. Группа файлов
        files_group = QtWidgets.QGroupBox("1. Загрузка файлов")
        files_layout = QtWidgets.QVBoxLayout(files_group)

        f1_row = QtWidgets.QHBoxLayout()
        self.btn_load1 = QtWidgets.QPushButton("Файл 1 (.fif / .set)...")
        self.btn_load1.setStyleSheet("background-color: #2563eb; color: white;")
        self.btn_load1.clicked.connect(lambda: self._on_browse_file(1))
        f1_row.addWidget(self.btn_load1)
        files_layout.addLayout(f1_row)

        self.lbl_info1 = QtWidgets.QLabel("Файл 1: не загружен")
        self.lbl_info1.setWordWrap(True)
        self.lbl_info1.setStyleSheet("color: #93c5fd; font-size: 11px;")
        files_layout.addWidget(self.lbl_info1)

        f2_row = QtWidgets.QHBoxLayout()
        self.btn_load2 = QtWidgets.QPushButton("Файл 2 (.fif / .set)...")
        self.btn_load2.setStyleSheet("background-color: #d97706; color: white;")
        self.btn_load2.clicked.connect(lambda: self._on_browse_file(2))
        f2_row.addWidget(self.btn_load2)
        files_layout.addLayout(f2_row)

        self.lbl_info2 = QtWidgets.QLabel("Файл 2: не загружен")
        self.lbl_info2.setWordWrap(True)
        self.lbl_info2.setStyleSheet("color: #fcd34d; font-size: 11px;")
        files_layout.addWidget(self.lbl_info2)

        left_layout.addWidget(files_group)

        # 2. Интерактивный сдвиг во времени (Time Shift)
        shift_group = QtWidgets.QGroupBox("2. Сдвиг во времени (Time Shift)")
        shift_layout = QtWidgets.QVBoxLayout(shift_group)

        # Выбор шага сдвига: 50 мс по умолчанию
        step_row = QtWidgets.QHBoxLayout()
        step_row.addWidget(QtWidgets.QLabel("Шаг:"))
        self.combo_shift_step = QtWidgets.QComboBox()
        self.combo_shift_step.addItems(["10 мс", "50 мс", "100 мс", "200 мс", "500 мс", "1000 мс"])
        self.combo_shift_step.setCurrentText("50 мс")
        self.combo_shift_step.currentIndexChanged.connect(self._on_shift_step_combo_changed)
        step_row.addWidget(self.combo_shift_step)
        shift_layout.addLayout(step_row)

        # Кнопки сдвига влево / вправо
        shift_btns_row = QtWidgets.QHBoxLayout()
        self.btn_shift_left = QtWidgets.QPushButton("◄ -50 мс")
        self.btn_shift_left.clicked.connect(lambda: self._shift_offset_ms(-self.shift_step_ms))

        self.btn_shift_right = QtWidgets.QPushButton("+50 мс ►")
        self.btn_shift_right.clicked.connect(lambda: self._shift_offset_ms(self.shift_step_ms))

        self.btn_shift_reset = QtWidgets.QPushButton("0 мс")
        self.btn_shift_reset.setProperty("class", "secondary")
        self.btn_shift_reset.setToolTip("Сбросить сдвиг в 0")
        self.btn_shift_reset.clicked.connect(self._reset_offset)

        shift_btns_row.addWidget(self.btn_shift_left)
        shift_btns_row.addWidget(self.btn_shift_right)
        shift_btns_row.addWidget(self.btn_shift_reset)
        shift_layout.addLayout(shift_btns_row)

        # Точный ввод сдвига в миллисекундах и секундах
        offset_input_row = QtWidgets.QHBoxLayout()
        offset_input_row.addWidget(QtWidgets.QLabel("Сдвиг Ф2:"))
        self.spin_offset_ms = QtWidgets.QDoubleSpinBox()
        self.spin_offset_ms.setRange(-3600000.0, 3600000.0)
        self.spin_offset_ms.setSingleStep(50.0)
        self.spin_offset_ms.setValue(0.0)
        self.spin_offset_ms.setSuffix(" мс")
        self.spin_offset_ms.valueChanged.connect(self._on_spin_offset_ms_changed)
        offset_input_row.addWidget(self.spin_offset_ms)
        shift_layout.addLayout(offset_input_row)

        # Строка подсказки текущего сдвига и автоподбор
        sub_shift_row = QtWidgets.QHBoxLayout()
        self.lbl_shift_status = QtWidgets.QLabel("Текущий: 0.0 мс (0.000 с)")
        self.lbl_shift_status.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold;")
        btn_auto_align = QtWidgets.QPushButton("⚡ Автосдвиг")
        btn_auto_align.setProperty("class", "secondary")
        btn_auto_align.setToolTip("Автоматически подобрать сдвиг (±1с) по максимуму корреляции")
        btn_auto_align.clicked.connect(self._auto_align_offset)
        sub_shift_row.addWidget(self.lbl_shift_status)
        sub_shift_row.addStretch()
        sub_shift_row.addWidget(btn_auto_align)
        shift_layout.addLayout(sub_shift_row)

        left_layout.addWidget(shift_group)

        # 3. Обрезание графиков (Trimming) & Корреляция
        trim_group = QtWidgets.QGroupBox("3. Обрезание & Корреляция (Trimming)")
        trim_layout = QtWidgets.QVBoxLayout(trim_group)

        self.cb_enable_trim = QtWidgets.QCheckBox("Включить обрезку графиков")
        self.cb_enable_trim.setChecked(False)
        self.cb_enable_trim.toggled.connect(self._on_toggle_trim)
        trim_layout.addWidget(self.cb_enable_trim)

        # Настройки обрезки Файла 1
        f1_trim_row = QtWidgets.QHBoxLayout()
        f1_trim_row.addWidget(QtWidgets.QLabel("Ф1: Старт:"))
        self.spin_f1_start = QtWidgets.QDoubleSpinBox()
        self.spin_f1_start.setRange(0.0, 100000.0)
        self.spin_f1_start.setValue(0.0)
        self.spin_f1_start.setSuffix(" с")
        self.spin_f1_start.valueChanged.connect(self._on_trim_params_changed)
        f1_trim_row.addWidget(self.spin_f1_start)

        f1_trim_row.addWidget(QtWidgets.QLabel("Конец:"))
        self.spin_f1_end = QtWidgets.QDoubleSpinBox()
        self.spin_f1_end.setRange(0.0, 100000.0)
        self.spin_f1_end.setValue(0.0)
        self.spin_f1_end.setSuffix(" с")
        self.spin_f1_end.valueChanged.connect(self._on_trim_params_changed)
        f1_trim_row.addWidget(self.spin_f1_end)
        trim_layout.addLayout(f1_trim_row)

        # Настройки обрезки Файла 2
        f2_trim_row = QtWidgets.QHBoxLayout()
        f2_trim_row.addWidget(QtWidgets.QLabel("Ф2: Старт:"))
        self.spin_f2_start = QtWidgets.QDoubleSpinBox()
        self.spin_f2_start.setRange(0.0, 100000.0)
        self.spin_f2_start.setValue(0.0)
        self.spin_f2_start.setSuffix(" с")
        self.spin_f2_start.valueChanged.connect(self._on_trim_params_changed)
        f2_trim_row.addWidget(self.spin_f2_start)

        f2_trim_row.addWidget(QtWidgets.QLabel("Конец:"))
        self.spin_f2_end = QtWidgets.QDoubleSpinBox()
        self.spin_f2_end.setRange(0.0, 100000.0)
        self.spin_f2_end.setValue(0.0)
        self.spin_f2_end.setSuffix(" с")
        self.spin_f2_end.valueChanged.connect(self._on_trim_params_changed)
        f2_trim_row.addWidget(self.spin_f2_end)
        trim_layout.addLayout(f2_trim_row)

        # Кнопка расчета корреляции по обрезанным кусочкам
        self.btn_calc_corr = QtWidgets.QPushButton("📊 Рассчитать корреляцию по кусочкам")
        self.btn_calc_corr.setProperty("class", "accent")
        self.btn_calc_corr.setStyleSheet("background-color: #059669; color: white; padding: 7px; font-size: 13px;")
        self.btn_calc_corr.clicked.connect(self._compute_correlation_on_trimmed)
        trim_layout.addWidget(self.btn_calc_corr)

        left_layout.addWidget(trim_group)

        # 4. Группа выбора каналов
        chan_group = QtWidgets.QGroupBox("4. Выбор каналов")
        chan_layout = QtWidgets.QVBoxLayout(chan_group)

        self.edit_chan_search = QtWidgets.QLineEdit()
        self.edit_chan_search.setPlaceholderText("Фильтр (напр. Fz, Cz, Pz)...")
        self.edit_chan_search.textChanged.connect(self._filter_channel_list)
        chan_layout.addWidget(self.edit_chan_search)

        self.list_channels = ChannelListWidget(app_window=self)
        self.list_channels.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_channels.itemChanged.connect(self._on_channel_checked_changed)
        chan_layout.addWidget(self.list_channels)

        chan_btns_row1 = QtWidgets.QHBoxLayout()
        self.btn_clear = QtWidgets.QPushButton("Снять все")
        self.btn_clear.setProperty("class", "secondary")
        self.btn_clear.setStyleSheet("background-color: #475569; color: #f8fafc;")
        self.btn_clear.clicked.connect(self._deselect_all_channels)

        self.btn_all_common = QtWidgets.QPushButton("Все общие")
        self.btn_all_common.setProperty("class", "secondary")
        self.btn_all_common.setStyleSheet("background-color: #334155; color: #93c5fd;")
        self.btn_all_common.clicked.connect(self._select_all_common_channels)

        chan_btns_row1.addWidget(self.btn_clear)
        chan_btns_row1.addWidget(self.btn_all_common)
        chan_layout.addLayout(chan_btns_row1)

        page_layout = QtWidgets.QHBoxLayout()
        page_layout.addWidget(QtWidgets.QLabel("Каналов в окне:"))
        self.combo_chans_per_page = QtWidgets.QComboBox()
        self.combo_chans_per_page.addItems(["8", "16", "24", "32", "Все"])
        self.combo_chans_per_page.setCurrentText("16")
        self.combo_chans_per_page.currentIndexChanged.connect(self._on_chans_per_page_changed)
        page_layout.addWidget(self.combo_chans_per_page)
        chan_layout.addLayout(page_layout)

        page_nav_layout = QtWidgets.QHBoxLayout()
        self.btn_prev_page = QtWidgets.QPushButton("▲ Пред.")
        self.btn_prev_page.clicked.connect(self._prev_channel_page)
        self.btn_next_page = QtWidgets.QPushButton("▼ След.")
        self.btn_next_page.clicked.connect(self._next_channel_page)
        page_nav_layout.addWidget(self.btn_prev_page)
        page_nav_layout.addWidget(self.btn_next_page)
        chan_layout.addLayout(page_nav_layout)

        self.lbl_page_info = QtWidgets.QLabel("Выбрано: 0 каналов")
        self.lbl_page_info.setAlignment(Qt.AlignCenter)
        self.lbl_page_info.setStyleSheet("color: #60a5fa; font-size: 11px; font-weight: bold;")
        chan_layout.addWidget(self.lbl_page_info)

        left_layout.addWidget(chan_group)

        # 5. Режим и масштаб
        vis_group = QtWidgets.QGroupBox("5. Режим и масштаб")
        vis_layout = QtWidgets.QVBoxLayout(vis_group)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Режим:"))
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItem("Наложение (Overlay)", "overlay")
        self.combo_mode.addItem("Разница (File1 - File2)", "diff")
        self.combo_mode.addItem("Раздельно (Split)", "split")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.combo_mode)
        vis_layout.addLayout(mode_layout)

        scale_layout = QtWidgets.QHBoxLayout()
        scale_layout.addWidget(QtWidgets.QLabel("Размах (мкВ):"))
        self.spin_scale = QtWidgets.QDoubleSpinBox()
        self.spin_scale.setRange(1.0, 5000.0)
        self.spin_scale.setValue(50.0)
        self.spin_scale.setSingleStep(10.0)
        self.spin_scale.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.spin_scale)

        btn_scale_down = QtWidgets.QPushButton("-")
        btn_scale_down.setFixedWidth(30)
        btn_scale_down.clicked.connect(lambda: self.spin_scale.setValue(max(5.0, self.spin_scale.value() / 1.5)))
        btn_scale_up = QtWidgets.QPushButton("+")
        btn_scale_up.setFixedWidth(30)
        btn_scale_up.clicked.connect(lambda: self.spin_scale.setValue(self.spin_scale.value() * 1.5))
        scale_layout.addWidget(btn_scale_down)
        scale_layout.addWidget(btn_scale_up)
        vis_layout.addLayout(scale_layout)

        dur_layout = QtWidgets.QHBoxLayout()
        dur_layout.addWidget(QtWidgets.QLabel("Окно (сек):"))
        self.combo_win_dur = QtWidgets.QComboBox()
        self.combo_win_dur.addItems(["1.0", "2.0", "5.0", "10.0", "20.0", "30.0"])
        self.combo_win_dur.setCurrentText("5.0")
        self.combo_win_dur.currentIndexChanged.connect(self._on_win_dur_changed)
        dur_layout.addWidget(self.combo_win_dur)
        vis_layout.addLayout(dur_layout)

        left_layout.addWidget(vis_group)

        # 6. Группа меток (Markers)
        markers_group = QtWidgets.QGroupBox("6. Метки файлов (Events)")
        markers_layout = QtWidgets.QVBoxLayout(markers_group)

        self.cb_show_markers = QtWidgets.QCheckBox("Отображать метки на графике")
        self.cb_show_markers.setChecked(True)
        self.cb_show_markers.toggled.connect(self._on_toggle_show_markers)
        markers_layout.addWidget(self.cb_show_markers)

        m_files_row = QtWidgets.QHBoxLayout()
        self.cb_markers1 = QtWidgets.QCheckBox("Метки Ф1 (0)")
        self.cb_markers1.setChecked(True)
        self.cb_markers1.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.cb_markers1.toggled.connect(self._on_toggle_marker_files)

        self.cb_markers2 = QtWidgets.QCheckBox("Метки Ф2 (0)")
        self.cb_markers2.setChecked(True)
        self.cb_markers2.setStyleSheet("color: #fb923c; font-weight: bold;")
        self.cb_markers2.toggled.connect(self._on_toggle_marker_files)

        m_files_row.addWidget(self.cb_markers1)
        m_files_row.addWidget(self.cb_markers2)
        markers_layout.addLayout(m_files_row)

        m_nav_row = QtWidgets.QHBoxLayout()
        btn_prev_marker = QtWidgets.QPushButton("◄ Пред. метка")
        btn_prev_marker.clicked.connect(self._goto_prev_marker)
        btn_next_marker = QtWidgets.QPushButton("След. метка ►")
        btn_next_marker.clicked.connect(self._goto_next_marker)
        m_nav_row.addWidget(btn_prev_marker)
        m_nav_row.addWidget(btn_next_marker)
        markers_layout.addLayout(m_nav_row)

        btn_list_markers = QtWidgets.QPushButton("Список меток...")
        btn_list_markers.setProperty("class", "secondary")
        btn_list_markers.clicked.connect(self._show_markers_dialog)
        markers_layout.addWidget(btn_list_markers)

        left_layout.addWidget(markers_group)
        left_layout.addStretch()

        scroll_area.setWidget(left_panel)
        main_layout.addWidget(scroll_area)

        # Правая область: График + Навигация по времени
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Верхняя панель легенды и статуса
        top_bar = QtWidgets.QHBoxLayout()
        self.lbl_legend1 = QtWidgets.QLabel("■ Файл 1 (Синий)")
        self.lbl_legend1.setStyleSheet("color: #60a5fa; font-weight: bold; font-size: 13px;")
        self.lbl_legend2 = QtWidgets.QLabel("■ Файл 2 (Оранжевый / Наложение)")
        self.lbl_legend2.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 13px;")
        self.lbl_legend_markers = QtWidgets.QLabel(" | Метки: Ф1 (голубой), Ф2 (оранжевый)")
        self.lbl_legend_markers.setStyleSheet("color: #38bdf8; font-size: 12px;")

        self.lbl_metrics = QtWidgets.QLabel("Метрики окна: -")
        self.lbl_metrics.setStyleSheet("color: #10b981; font-weight: bold;")

        top_bar.addWidget(self.lbl_legend1)
        top_bar.addWidget(self.lbl_legend2)
        top_bar.addWidget(self.lbl_legend_markers)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_metrics)
        right_layout.addLayout(top_bar)

        # Холст графика
        self.canvas = MplCanvas(self, width=10, height=8)
        right_layout.addWidget(self.canvas, stretch=1)

        # Нижняя панель навигации (Скролл во времени)
        time_nav_panel = QtWidgets.QWidget()
        time_nav_layout = QtWidgets.QVBoxLayout(time_nav_panel)
        time_nav_layout.setContentsMargins(4, 4, 4, 4)
        time_nav_layout.setSpacing(4)

        slider_row = QtWidgets.QHBoxLayout()
        self.lbl_cur_time = QtWidgets.QLabel("0.0 с")
        self.lbl_cur_time.setFixedWidth(60)
        self.lbl_cur_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider_row.addWidget(self.lbl_cur_time)

        self.time_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.time_slider)

        self.lbl_total_time = QtWidgets.QLabel("/ 0.0 с")
        self.lbl_total_time.setFixedWidth(70)
        slider_row.addWidget(self.lbl_total_time)
        time_nav_layout.addLayout(slider_row)

        buttons_row = QtWidgets.QHBoxLayout()
        btn_start = QtWidgets.QPushButton("|< Начало")
        btn_start.clicked.connect(self._goto_start)
        btn_bwd_big = QtWidgets.QPushButton("<< -10с")
        btn_bwd_big.clicked.connect(lambda: self._shift_time(-10.0))
        btn_bwd = QtWidgets.QPushButton("< -1с")
        btn_bwd.clicked.connect(lambda: self._shift_time(-1.0))
        btn_fwd = QtWidgets.QPushButton("+1с >")
        btn_fwd.clicked.connect(lambda: self._shift_time(1.0))
        btn_fwd_big = QtWidgets.QPushButton("+10с >>")
        btn_fwd_big.clicked.connect(lambda: self._shift_time(10.0))
        btn_end = QtWidgets.QPushButton("Конец >|")
        btn_end.clicked.connect(self._goto_end)

        buttons_row.addWidget(btn_start)
        buttons_row.addWidget(btn_bwd_big)
        buttons_row.addWidget(btn_bwd)
        buttons_row.addWidget(btn_fwd)
        buttons_row.addWidget(btn_fwd_big)
        buttons_row.addWidget(btn_end)
        buttons_row.addStretch()

        buttons_row.addWidget(QtWidgets.QLabel("Перейти на (с):"))
        self.spin_direct_time = QtWidgets.QDoubleSpinBox()
        self.spin_direct_time.setRange(0.0, 100000.0)
        self.spin_direct_time.setSingleStep(5.0)
        self.spin_direct_time.setValue(0.0)
        btn_go_time = QtWidgets.QPushButton("Перейти")
        btn_go_time.clicked.connect(lambda: self._set_time(self.spin_direct_time.value()))
        buttons_row.addWidget(self.spin_direct_time)
        buttons_row.addWidget(btn_go_time)

        time_nav_layout.addLayout(buttons_row)
        right_layout.addWidget(time_nav_panel)

        main_layout.addWidget(right_panel, stretch=1)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Готово к работе. Загрузите файлы .fif или .set для сравнения.")

    # ------------------ Загрузка данных ------------------

    def _on_browse_file(self, file_idx: int):
        last_dir = str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Выберите Файл {file_idx} (MNE .fif или EEGLAB .set)",
            last_dir,
            "EEG Files (*.fif *.fif.gz *.set);;FIF files (*.fif *.fif.gz);;EEGLAB files (*.set);;All Files (*)"
        )
        if path:
            self.load_file(file_idx, path)

    def load_file(self, file_idx: int, path: str):
        try:
            self.status_bar.showMessage(f"Загрузка файла {file_idx}: {Path(path).name}...")
            QtWidgets.QApplication.processEvents()

            raw = load_eeg_file(path)
            markers = extract_raw_markers(raw, file_path=path)

            if file_idx == 1:
                self.raw1 = raw
                self.file1_path = path
                self.markers1 = markers
                dur = raw.times[-1]
                sf = raw.info["sfreq"]
                n_ch = len(raw.ch_names)
                n_m = len(markers)
                self.lbl_info1.setText(f"<b>Файл 1:</b> {Path(path).name}<br>Fs={sf:.1f}Hz, {n_ch} кан., {dur:.1f}с | Меток: {n_m}")
                self.cb_markers1.setText(f"Метки Ф1 ({n_m})")
                self.spin_f1_start.setMaximum(dur)
                self.spin_f1_end.setMaximum(dur)
            else:
                self.raw2 = raw
                self.file2_path = path
                self.markers2 = markers
                dur = raw.times[-1]
                sf = raw.info["sfreq"]
                n_ch = len(raw.ch_names)
                n_m = len(markers)
                self.lbl_info2.setText(f"<b>Файл 2:</b> {Path(path).name}<br>Fs={sf:.1f}Hz, {n_ch} кан., {dur:.1f}с | Меток: {n_m}")
                self.cb_markers2.setText(f"Метки Ф2 ({n_m})")
                self.spin_f2_start.setMaximum(dur)
                self.spin_f2_end.setMaximum(dur)

            self._update_time_slider_range()
            self._rebuild_channel_list()
            self.redraw()
            self.status_bar.showMessage(f"Файл {file_idx} успешно загружен: {Path(path).name} (Меток: {len(markers)})", 5000)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить файл:\n{str(e)}")
            self.status_bar.showMessage(f"Ошибка загрузки файла {file_idx}")

    # ------------------ Управление сдвигом (Time Shift) ------------------

    def _on_shift_step_combo_changed(self):
        text = self.combo_shift_step.currentText()
        val_ms = float(text.replace(" мс", "").strip())
        self.shift_step_ms = val_ms
        self.btn_shift_left.setText(f"◄ -{int(val_ms)} мс")
        self.btn_shift_right.setText(f"+{int(val_ms)} мс ►")

    def _shift_offset_ms(self, delta_ms: float):
        new_ms = (self.time_offset2 * 1000.0) + delta_ms
        self._set_offset_ms(new_ms)

    def _reset_offset(self):
        self._set_offset_ms(0.0)

    def _set_offset_ms(self, offset_ms: float):
        self.time_offset2 = offset_ms / 1000.0
        self.spin_offset_ms.blockSignals(True)
        self.spin_offset_ms.setValue(offset_ms)
        self.spin_offset_ms.blockSignals(False)
        self.lbl_shift_status.setText(f"Текущий: {offset_ms:+.1f} мс ({self.time_offset2:+.3f} с)")
        self._update_time_slider_range()
        self.redraw()

    def _on_spin_offset_ms_changed(self, val: float):
        self._set_offset_ms(val)

    def _auto_align_offset(self):
        """Автоматический подбор сдвига по максимуму кросс-корреляции."""
        if self.raw1 is None or self.raw2 is None:
            QtWidgets.QMessageBox.warning(self, "Внимание", "Загрузите оба файла для автовыравнивания.")
            return

        common = [c for c in self.selected_channels if c in self.raw1.ch_names and c in self.raw2.ch_names]
        if not common:
            QtWidgets.QMessageBox.warning(self, "Внимание", "Выберите хотя бы один общий канал.")
            return

        self.status_bar.showMessage("Автоподбор сдвига (поиск максимума корреляции)...")
        QtWidgets.QApplication.processEvents()

        # Берем фрагмент 20 секунд из центра
        dur = min(self.raw1.times[-1], self.raw2.times[-1])
        t_center = dur / 2.0
        t0 = max(0.0, t_center - 10.0)
        t1 = min(dur, t_center + 10.0)

        sf1 = self.raw1.info["sfreq"]
        sf2 = self.raw2.info["sfreq"]
        s0_1 = int(round(t0 * sf1))
        s1_1 = int(round(t1 * sf1))

        best_lag_ms = 0.0
        best_r = -1.0

        # Поиск в диапазоне ±1.0 сек с шагом 5 мс
        lags_ms = np.arange(-1000.0, 1005.0, 5.0)

        ch_idx1 = [self.raw1.ch_names.index(c) for c in common[:5]]  # берем первые до 5 каналов для скорости
        ch_idx2 = [self.raw2.ch_names.index(c) for c in common[:5]]

        d1 = self.raw1._data[ch_idx1, s0_1:s1_1]
        t_base = np.arange(s0_1, s1_1) / sf1

        for lag in lags_ms:
            t0_2 = t0 - (lag / 1000.0)
            t1_2 = t1 - (lag / 1000.0)
            s0_2 = int(round(t0_2 * sf2))
            s1_2 = int(round(t1_2 * sf2))

            if s0_2 < 0 or s1_2 > self.raw2.n_times:
                continue

            d2 = self.raw2._data[ch_idx2, s0_2:s1_2]
            t2 = (np.arange(s0_2, s1_2) / sf2) + (lag / 1000.0)

            # Интерполяция d2 на t_base
            r_list = []
            for k in range(len(ch_idx1)):
                d2_interp = np.interp(t_base, t2, d2[k], left=np.nan, right=np.nan)
                m = ~(np.isnan(d1[k]) | np.isnan(d2_interp))
                if np.sum(m) > 100:
                    c = np.corrcoef(d1[k, m], d2_interp[m])[0, 1]
                    r_list.append(c)

            if r_list:
                mean_r = np.mean(r_list)
                if mean_r > best_r:
                    best_r = mean_r
                    best_lag_ms = lag

        self._set_offset_ms(best_lag_ms)
        self.status_bar.showMessage(f"Автовыравнивание завершено! Найден лучший сдвиг: {best_lag_ms:+.1f} мс (r = {best_r:.3f})", 6000)

    # ------------------ Обрезание & Расчет корреляции (Trimming) ------------------

    def _on_toggle_trim(self, checked: bool):
        self.trim_enabled = checked
        self.redraw()

    def _on_trim_params_changed(self):
        self.trim_f1_start = self.spin_f1_start.value()
        self.trim_f1_end = self.spin_f1_end.value()
        self.trim_f2_start = self.spin_f2_start.value()
        self.trim_f2_end = self.spin_f2_end.value()
        if self.trim_enabled:
            self.redraw()

    def _compute_correlation_on_trimmed(self):
        """Вычисляет корреляцию между файлами на обрезанных кусочках."""
        if self.raw1 is None or self.raw2 is None:
            QtWidgets.QMessageBox.warning(self, "Внимание", "Необходимо загрузить оба файла (Файл 1 и Файл 2).")
            return

        common_channels = [c for c in self.selected_channels if c in self.raw1.ch_names and c in self.raw2.ch_names]
        if not common_channels:
            QtWidgets.QMessageBox.warning(self, "Внимание", "Нет выбранных общих каналов для расчета.")
            return

        self.status_bar.showMessage("Расчет корреляции по обрезанным кусочкам...")
        QtWidgets.QApplication.processEvents()

        dur1 = self.raw1.times[-1]
        dur2 = self.raw2.times[-1]

        # Временные интервалы с учетом обрезки
        if self.trim_enabled:
            start1 = self.trim_f1_start
            stop1 = max(start1, dur1 - self.trim_f1_end)

            start2 = self.trim_f2_start
            stop2 = max(start2, dur2 - self.trim_f2_end)
        else:
            start1, stop1 = 0.0, dur1
            start2, stop2 = 0.0, dur2

        # С учетом time_offset2 для Файла 2 приводим к общей временной оси
        # Совмещенный интервал пересечения
        t_global_start = max(start1, start2 + self.time_offset2)
        t_global_stop = min(stop1, stop2 + self.time_offset2)

        if t_global_stop <= t_global_start + 0.1:
            QtWidgets.QMessageBox.warning(
                self, "Ошибка пересечения",
                f"После обрезки и сдвига интервалы не пересекаются!\n"
                f"Интервал Ф1: [{start1:.1f} .. {stop1:.1f}] с\n"
                f"Интервал Ф2 (со сдвигом {self.time_offset2:+.3f}с): [{start2 + self.time_offset2:.1f} .. {stop2 + self.time_offset2:.1f}] с"
            )
            return

        sf1 = self.raw1.info["sfreq"]
        sf2 = self.raw2.info["sfreq"]

        # Выбираем сетку времени с частотой sf1
        target_sfreq = sf1
        n_pts = int(round((t_global_stop - t_global_start) * target_sfreq))
        t_grid = np.linspace(t_global_start, t_global_stop, n_pts)

        # Вырезаем данные Файла 1
        s0_1 = int(round(t_global_start * sf1))
        s1_1 = s0_1 + n_pts
        s1_1 = min(s1_1, self.raw1.n_times)
        actual_pts = s1_1 - s0_1
        t_grid = t_grid[:actual_pts]

        ch_indices1 = [self.raw1.ch_names.index(c) for c in common_channels]
        ch_indices2 = [self.raw2.ch_names.index(c) for c in common_channels]

        data1 = self.raw1._data[ch_indices1, s0_1:s1_1]

        # Вырезаем данные Файла 2 с учетом сдвига
        t2_start_native = t_global_start - self.time_offset2
        t2_stop_native = t_global_stop - self.time_offset2
        s0_2 = max(0, int(round(t2_start_native * sf2)))
        s1_2 = min(self.raw2.n_times, int(round(t2_stop_native * sf2)))

        data2_raw = self.raw2._data[ch_indices2, s0_2:s1_2]
        t2_raw = (np.arange(s0_2, s1_2) / sf2) + self.time_offset2

        # Расчет Pearson r и RMSE по каналам
        channel_stats = []
        all_r = []
        all_rmse = []

        for i, ch_name in enumerate(common_channels):
            v1 = data1[i]
            # Интерполируем v2 на t_grid
            v2 = np.interp(t_grid, t2_raw, data2_raw[i], left=np.nan, right=np.nan)
            mask = ~(np.isnan(v1) | np.isnan(v2))

            if np.sum(mask) > 10:
                x = v1[mask]
                y = v2[mask]
                sx = np.std(x)
                sy = np.std(y)
                r_val = float(np.corrcoef(x, y)[0, 1]) if (sx > 1e-12 and sy > 1e-12) else 0.0
                rmse_val = float(np.sqrt(np.mean((x - y) ** 2)) * 1e6)  # в мкВ
            else:
                r_val = 0.0
                rmse_val = 0.0

            channel_stats.append({
                "channel": ch_name,
                "r": r_val,
                "rmse": rmse_val,
            })
            all_r.append(r_val)
            all_rmse.append(rmse_val)

        stats_payload = {
            "mean_r": float(np.mean(all_r)),
            "min_r": float(np.min(all_r)),
            "max_r": float(np.max(all_r)),
            "mean_rmse": float(np.mean(all_rmse)),
            "eff_duration": float(t_global_stop - t_global_start),
            "n_samples": actual_pts,
            "channel_stats": channel_stats,
        }

        self.status_bar.showMessage(
            f"Корреляция рассчитана: Средн. r = {stats_payload['mean_r']:.4f} | RMSE = {stats_payload['mean_rmse']:.2f} мкВ",
            8000
        )

        dlg = CorrelationDialog(stats_payload, parent=self)
        dlg.exec_()

    # ------------------ Пагинация и выбор каналов ------------------

    def _rebuild_channel_list(self):
        self.list_channels.blockSignals(True)
        self.list_channels.clear()

        all_channels: List[str] = []
        chans1 = set(self.raw1.ch_names) if self.raw1 is not None else set()
        chans2 = set(self.raw2.ch_names) if self.raw2 is not None else set()

        if self.raw1 and self.raw2:
            common = [c for c in self.raw1.ch_names if c in chans2]
            only1 = [c for c in self.raw1.ch_names if c not in chans2]
            only2 = [c for c in self.raw2.ch_names if c not in chans1]
            all_channels = common + only1 + only2
        elif self.raw1:
            all_channels = list(self.raw1.ch_names)
        elif self.raw2:
            all_channels = list(self.raw2.ch_names)

        common_set = chans1.intersection(chans2) if (self.raw1 and self.raw2) else set(all_channels)

        for ch in all_channels:
            item = QtWidgets.QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

            if ch in common_set:
                item.setCheckState(Qt.Checked)
                item.setForeground(QtGui.QBrush(QtGui.QColor("#93c5fd")))
            else:
                item.setCheckState(Qt.Unchecked)
                item.setForeground(QtGui.QBrush(QtGui.QColor("#64748b")))

            self.list_channels.addItem(item)

        self.list_channels.blockSignals(False)
        self._update_selected_channels_from_ui()

    def _filter_channel_list(self, filter_text: str):
        filter_text = filter_text.strip().lower()
        for i in range(self.list_channels.count()):
            item = self.list_channels.item(i)
            match = (filter_text in item.text().lower())
            item.setHidden(not match)

    def _on_channel_checked_changed(self, item):
        self._update_selected_channels_from_ui()
        self.current_page = 0
        self.redraw()

    def _update_selected_channels_from_ui(self):
        self.selected_channels = []
        for i in range(self.list_channels.count()):
            item = self.list_channels.item(i)
            if item.checkState() == Qt.Checked:
                self.selected_channels.append(item.text())
        self._update_page_info()

    def solo_channel(self, ch_name: str):
        self.list_channels.blockSignals(True)
        for i in range(self.list_channels.count()):
            item = self.list_channels.item(i)
            if item.text() == ch_name:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
        self.list_channels.blockSignals(False)

        self._update_selected_channels_from_ui()
        self.current_page = 0
        self.redraw()

    def set_visible_filter_channels_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        self.list_channels.blockSignals(True)
        for i in range(self.list_channels.count()):
            item = self.list_channels.item(i)
            if not item.isHidden():
                item.setCheckState(state)
        self.list_channels.blockSignals(False)

        self._update_selected_channels_from_ui()
        self.current_page = 0
        self.redraw()

    def _select_all_common_channels(self):
        chans1 = set(self.raw1.ch_names) if self.raw1 is not None else set()
        chans2 = set(self.raw2.ch_names) if self.raw2 is not None else set()
        common = chans1.intersection(chans2) if (self.raw1 and self.raw2) else (chans1 or chans2)

        self.list_channels.blockSignals(True)
        for i in range(self.list_channels.count()):
            item = self.list_channels.item(i)
            if item.text() in common:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
        self.list_channels.blockSignals(False)

        self._update_selected_channels_from_ui()
        self.current_page = 0
        self.redraw()

    def _deselect_all_channels(self):
        self.list_channels.blockSignals(True)
        for i in range(self.list_channels.count()):
            self.list_channels.item(i).setCheckState(Qt.Unchecked)
        self.list_channels.blockSignals(False)

        self._update_selected_channels_from_ui()
        self.current_page = 0
        self.redraw()

    def _on_chans_per_page_changed(self):
        txt = self.combo_chans_per_page.currentText()
        self.channels_per_page = 9999 if txt == "Все" else int(txt)
        self.current_page = 0
        self._update_page_info()
        self.redraw()

    def _prev_channel_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._update_page_info()
            self.redraw()

    def _next_channel_page(self):
        max_page = max(0, (len(self.selected_channels) - 1) // self.channels_per_page)
        if self.current_page < max_page:
            self.current_page += 1
            self._update_page_info()
            self.redraw()

    def _update_page_info(self):
        total_ch = len(self.selected_channels)
        if total_ch == 0:
            self.lbl_page_info.setText("Выбрано: 0 каналов")
            return
        if self.channels_per_page >= total_ch:
            self.lbl_page_info.setText(f"Выбрано: {total_ch} кан. (Все на экране)")
        else:
            max_page = max(0, (total_ch - 1) // self.channels_per_page)
            self.lbl_page_info.setText(f"Стр. {self.current_page + 1}/{max_page + 1} (Всего: {total_ch})")

    def _get_current_visible_channels(self) -> List[str]:
        if not self.selected_channels:
            return []
        start = self.current_page * self.channels_per_page
        end = start + self.channels_per_page
        return self.selected_channels[start:end]

    # ------------------ Навигация и контроллеры ------------------

    def _get_max_duration(self) -> float:
        dur1 = self.raw1.times[-1] if self.raw1 is not None else 0.0
        dur2 = (self.raw2.times[-1] + abs(self.time_offset2)) if self.raw2 is not None else 0.0
        return max(dur1, dur2)

    def _update_time_slider_range(self):
        max_dur = self._get_max_duration()
        self.lbl_total_time.setText(f"/ {max_dur:.1f} с")
        self.spin_direct_time.setMaximum(max_dur)

        steps = int(max(1, max_dur * 10))
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, steps)
        cur_step = int(self.time_pos * 10)
        self.time_slider.setValue(min(cur_step, steps))
        self.time_slider.blockSignals(False)

    def _on_slider_changed(self, value: int):
        self.time_pos = value / 10.0
        self.lbl_cur_time.setText(f"{self.time_pos:.1f} с")
        self.spin_direct_time.blockSignals(True)
        self.spin_direct_time.setValue(self.time_pos)
        self.spin_direct_time.blockSignals(False)
        self.redraw()

    def _set_time(self, new_time: float):
        max_time = max(0.0, self._get_max_duration() - self.win_duration)
        self.time_pos = float(np.clip(new_time, 0.0, max(0.0, max_time)))
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(self.time_pos * 10))
        self.time_slider.blockSignals(False)
        self.lbl_cur_time.setText(f"{self.time_pos:.1f} с")
        self.redraw()

    def _shift_time(self, delta_s: float):
        self._set_time(self.time_pos + delta_s)

    def _goto_start(self):
        self._set_time(0.0)

    def _goto_end(self):
        self._set_time(self._get_max_duration() - self.win_duration)

    def _on_mode_changed(self):
        self.display_mode = self.combo_mode.currentData()
        if self.display_mode == "diff":
            self.lbl_legend1.setText("■ Разница (Файл 1 - Файл 2)")
            self.lbl_legend1.setStyleSheet("color: #34d399; font-weight: bold;")
            self.lbl_legend2.setText("")
        elif self.display_mode == "split":
            self.lbl_legend1.setText("■ Файл 1 (Сверху)")
            self.lbl_legend1.setStyleSheet("color: #60a5fa; font-weight: bold;")
            self.lbl_legend2.setText("■ Файл 2 (Снизу)")
            self.lbl_legend2.setStyleSheet("color: #f59e0b; font-weight: bold;")
        else:
            self.lbl_legend1.setText("■ Файл 1 (Синий)")
            self.lbl_legend1.setStyleSheet("color: #60a5fa; font-weight: bold;")
            self.lbl_legend2.setText("■ Файл 2 (Оранжевый / Наложение)")
            self.lbl_legend2.setStyleSheet("color: #f59e0b; font-weight: bold;")
        self.redraw()

    def _on_scale_changed(self, val: float):
        self.scale_uv = float(val)
        self.redraw()

    def _on_win_dur_changed(self):
        self.win_duration = float(self.combo_win_dur.currentText())
        self._update_time_slider_range()
        self.redraw()

    # ------------------ Метки (Markers) ------------------

    def _on_toggle_show_markers(self, checked: bool):
        self.show_markers = checked
        self.redraw()

    def _on_toggle_marker_files(self):
        self.show_markers1 = self.cb_markers1.isChecked()
        self.show_markers2 = self.cb_markers2.isChecked()
        self.redraw()

    def _get_active_markers(self) -> List[Tuple[float, float, str, str]]:
        active = []
        if not self.show_markers:
            return active

        if self.show_markers1:
            for m in self.markers1:
                active.append((m[0], m[1], m[2], "Ф1"))

        if self.show_markers2:
            for m in self.markers2:
                active.append((m[0] + self.time_offset2, m[1], m[2], "Ф2"))

        active.sort(key=lambda x: x[0])
        return active

    def _goto_next_marker(self):
        markers = self._get_active_markers()
        if not markers:
            self.status_bar.showMessage("Нет активных меток", 3000)
            return
        cur_center = self.time_pos + (self.win_duration / 2.0)
        next_m = next((m for m in markers if m[0] > cur_center + 0.05), markers[0])
        self._set_time(max(0.0, next_m[0] - self.win_duration / 2.0))

    def _goto_prev_marker(self):
        markers = self._get_active_markers()
        if not markers:
            self.status_bar.showMessage("Нет активных меток", 3000)
            return
        cur_center = self.time_pos + (self.win_duration / 2.0)
        prev_m = next((m for m in reversed(markers) if m[0] < cur_center - 0.05), markers[-1])
        self._set_time(max(0.0, prev_m[0] - self.win_duration / 2.0))

    def _show_markers_dialog(self):
        dlg = MarkersDialog(self, parent=self)
        dlg.exec_()

    # ------------------ Отрисовка ЭЭГ графиков и меток ------------------

    def redraw(self):
        """Главный цикл отрисовки сигналов, зон обрезки и меток на Canvas."""
        ax = self.canvas.ax
        ax.clear()

        visible_chans = self._get_current_visible_channels()

        if self.raw1 is None and self.raw2 is None:
            ax.text(0.5, 0.5, "Загрузите файлы для сравнения\n(Файл 1 и/или Файл 2 слева)",
                    color="#94a3b8", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        if not visible_chans:
            ax.text(0.5, 0.5, "Все каналы отключены.\nКликните по названию нужного канала слева, чтобы включить его.",
                    color="#94a3b8", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        t_start = self.time_pos
        t_stop = t_start + self.win_duration
        scale_v = self.scale_uv * 1e-6
        n_ch = len(visible_chans)

        y_ticks = []
        y_ticklabels = []

        corrs = []
        rmses = []

        for idx, ch_name in enumerate(visible_chans):
            y_base = (n_ch - 1 - idx) * scale_v
            y_ticks.append(y_base)
            y_ticklabels.append(ch_name)

            ax.axhline(y_base, color="#27272a", linestyle="--", linewidth=0.8)

            d1 = None
            d2 = None
            t1 = None
            t2 = None

            if self.raw1 is not None and ch_name in self.raw1.ch_names:
                sf1 = self.raw1.info["sfreq"]
                s_start1 = int(round(t_start * sf1))
                s_stop1 = int(round(t_stop * sf1))
                max_s1 = self.raw1.n_times
                if s_start1 < max_s1:
                    s_stop1 = min(s_stop1, max_s1)
                    ch_idx1 = self.raw1.ch_names.index(ch_name)
                    d1 = self.raw1._data[ch_idx1, s_start1:s_stop1]
                    t1 = np.arange(s_start1, s_stop1) / sf1

            if self.raw2 is not None and ch_name in self.raw2.ch_names:
                sf2 = self.raw2.info["sfreq"]
                t_start2 = t_start - self.time_offset2
                t_stop2 = t_stop - self.time_offset2
                s_start2 = int(round(t_start2 * sf2))
                s_stop2 = int(round(t_stop2 * sf2))
                max_s2 = self.raw2.n_times
                if s_start2 >= 0 and s_start2 < max_s2:
                    s_stop2 = min(s_stop2, max_s2)
                    ch_idx2 = self.raw2.ch_names.index(ch_name)
                    d2 = self.raw2._data[ch_idx2, s_start2:s_stop2]
                    t2 = (np.arange(s_start2, s_stop2) / sf2) + self.time_offset2

            if self.display_mode == "overlay":
                if d1 is not None and len(d1) > 0:
                    ax.plot(t1, d1 + y_base, color="#3b82f6", linewidth=1.1, alpha=0.9)
                if d2 is not None and len(d2) > 0:
                    ax.plot(t2, d2 + y_base, color="#f97316", linewidth=1.1, alpha=0.85)

            elif self.display_mode == "diff":
                if d1 is not None and d2 is not None and len(d1) > 0 and len(d2) > 0:
                    d2_interp = np.interp(t1, t2, d2, left=np.nan, right=np.nan)
                    diff = d1 - d2_interp
                    valid_mask = ~np.isnan(diff)
                    if np.any(valid_mask):
                        ax.plot(t1[valid_mask], diff[valid_mask] + y_base, color="#10b981", linewidth=1.2)

            elif self.display_mode == "split":
                if d1 is not None and len(d1) > 0:
                    ax.plot(t1, d1 + y_base + (0.25 * scale_v), color="#3b82f6", linewidth=1.0, alpha=0.9)
                if d2 is not None and len(d2) > 0:
                    ax.plot(t2, d2 + y_base - (0.25 * scale_v), color="#f97316", linewidth=1.0, alpha=0.9)

            if d1 is not None and d2 is not None and len(d1) > 10 and len(d2) > 10:
                try:
                    d2_interp = np.interp(t1, t2, d2, left=np.nan, right=np.nan)
                    m = ~(np.isnan(d1) | np.isnan(d2_interp))
                    if np.sum(m) > 10:
                        v1 = d1[m]
                        v2 = d2_interp[m]
                        std1, std2 = np.std(v1), np.std(v2)
                        if std1 > 1e-12 and std2 > 1e-12:
                            r = np.corrcoef(v1, v2)[0, 1]
                            corrs.append(r)
                        rmse_val = np.sqrt(np.mean((v1 - v2) ** 2)) * 1e6
                        rmses.append(rmse_val)
                except Exception:
                    pass

        # ------------------ Визуализация ОБРЕЗКИ (Trimming) ------------------
        y_min = -scale_v
        y_max = n_ch * scale_v

        if self.trim_enabled and (self.raw1 or self.raw2):
            dur1 = self.raw1.times[-1] if self.raw1 else 0.0
            dur2 = (self.raw2.times[-1] + self.time_offset2) if self.raw2 else 0.0

            # Зона отсечения начала
            cutoff_start = max(self.trim_f1_start, self.trim_f2_start + self.time_offset2)
            if cutoff_start > 0:
                ax.axvspan(0.0, cutoff_start, color="#dc2626", alpha=0.12, hatch="//")
                ax.axvline(cutoff_start, color="#ef4444", linestyle=":", linewidth=1.2)
                if cutoff_start >= t_start and cutoff_start <= t_stop:
                    ax.text(cutoff_start, y_max * 0.95, " [Обрезано]", color="#ef4444", fontsize=9, rotation=90)

            # Зона отсечения конца
            cutoff_end1 = dur1 - self.trim_f1_end if self.trim_f1_end > 0 else dur1
            cutoff_end2 = dur2 - self.trim_f2_end if self.trim_f2_end > 0 else dur2
            cutoff_stop = min(cutoff_end1, cutoff_end2)
            total_max = self._get_max_duration()

            if cutoff_stop < total_max:
                ax.axvspan(cutoff_stop, total_max, color="#dc2626", alpha=0.12, hatch="//")
                ax.axvline(cutoff_stop, color="#ef4444", linestyle=":", linewidth=1.2)
                if cutoff_stop >= t_start and cutoff_stop <= t_stop:
                    ax.text(cutoff_stop, y_max * 0.95, " [Обрезано]", color="#ef4444", fontsize=9, rotation=90)

        # ------------------ Отрисовка МЕТОК ------------------
        active_markers = self._get_active_markers()
        visible_markers_in_window = [
            m for m in active_markers
            if (m[0] + max(0.0, m[1])) >= t_start and m[0] <= t_stop
        ]

        for m_onset, m_dur, m_desc, m_fid in visible_markers_in_window:
            color = "#38bdf8" if m_fid == "Ф1" else "#fb923c"
            linestyle = "--" if m_fid == "Ф1" else "-."
            ax.axvline(m_onset, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.9)
            if m_dur > 0:
                ax.axvspan(m_onset, m_onset + m_dur, color=color, alpha=0.15)

            label_text = f"[{m_fid}] {m_desc}"
            ax.text(
                m_onset, y_max - (0.05 * scale_v), label_text,
                color=color, fontsize=9, fontweight="bold", rotation=90,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="square,pad=0.2", facecolor="#0f172a", edgecolor=color, alpha=0.85)
            )

        # Настройка осей и оформления
        ax.set_xlim(t_start, t_stop)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels, fontsize=10, color="#f1f5f9", fontweight="bold")
        ax.set_xlabel("Время (секунды)", fontsize=11, color="#cbd5e1", labelpad=8)
        ax.tick_params(axis="x", colors="#94a3b8", labelsize=10)
        ax.tick_params(axis="y", colors="#94a3b8")
        ax.grid(True, axis="x", color="#27272a", linestyle=":", linewidth=0.8)

        for spine in ax.spines.values():
            spine.set_color("#334155")

        shift_str = f"Сдвиг Ф2: {self.time_offset2*1000.0:+.0f}мс"
        if corrs:
            mean_r = np.mean(corrs)
            mean_rmse = np.mean(rmses)
            self.lbl_metrics.setText(f"{shift_str} | Окно: r = {mean_r:.3f} | RMSE = {mean_rmse:.2f} мкВ")
        else:
            self.lbl_metrics.setText(f"{shift_str} | Окно [{t_start:.1f} - {t_stop:.1f}s]")

        self.canvas.figure.tight_layout()
        self.canvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    file1 = sys.argv[1] if len(sys.argv) > 1 else None
    file2 = sys.argv[2] if len(sys.argv) > 2 else None

    window = EEGCompareApp(file1=file1, file2=file2)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
