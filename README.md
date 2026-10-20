# 🧠 Modular EEG-fMRI Cleaning Pipeline: Step 01 to Step 12

**High-precision modular software pipeline for concurrent EEG-fMRI gradient artifact removal (Bergen AAS), BCG (ballistocardiogram) OBS cleaning, in-place ASR burst repair, and Picard ICA with ICLabel auto-classification and quantitative alpha rhythm preservation control.**

---

## 📋 Pipeline Architecture & Workflow

The pipeline is split into two major production stages (**P01** and **P02**), with single-step modular runners and orchestrators:

```
Raw EEG (5000 Hz) + fMRI Motion (rp_*.txt)
  │
  ▼ [Stage P01: cleanP01.py / run_all.py (Steps 01-07)]
  ├─ 01: detect MRI sessions & match SPM motion files
  ├─ 02: sub-sample slice phase offset detection
  ├─ 03: dummy scan trimming & volume alignment
  ├─ 04: Optuna Bayesian tuning of Bergen AAS parameters
  ├─ 05: 96-channel Bergen AAS cleaning (5000 Hz)
  ├─ 06: Welch PSD spectral analysis & alpha preservation
  ├─ 07: Bergen HTML QC report
  └─ P01 Resample: 5000 Hz ➔ 250 Hz (data/<sid>/derivatives/02_resampled250/)
  │
  ▼ [Stage P02: cleanP02.py (Steps 08, 10, 11)]
  ├─ 08: fMRIB Optimal Basis Set (OBS) BCG removal (03_bcg/<seg>_bcg_clean.fif)
  ├─ 10: Fast Two-Phase optimization of clean_rawdata, ASR & ICLabel thresholds
  ├─ 11: Final cleaning: in-place ASR reconstruction + Picard ICA + ICLabel rejection
  └─ 12: Multi-stage interactive HTML report dashboard
  │
  ▼
Clean 250 Hz EEG Dataset: .fif (MNE) and .set (EEGLAB)
```

---

## 🛡️ Channel Preservation & Gentle Cleaning Policy (New)

Previous versions of EEG cleaning algorithms frequently over-removed channels using aggressive neighbor-correlation criteria (`ChannelCriterion >= 0.80`) or over-rejected 60–80% of ICA components, destroying brain variance and alpha rhythm.

**The pipeline now adopts a strict gentle-cleaning protocol:**
1. **`channel_crit: "off"` by default**: The algorithm does **not** delete channels based on spatial correlation. Channels are preserved intact across all 95 electrodes.
2. **PCA Dimensionality Reduction (`ica_pca_max: 35`)**: Stabilizes ICA on ~95 channels by projecting data into the top 35 principal components prior to decomposition. This completely prevents component splitting and keeps independent sources stable.
3. **Low-Pass Filter (`filter.lp: 45.0 Hz`)**: Effectively cuts the mechanical helium-pump vibration harmonic comb (44–100 Hz), which Bergen AAS cannot eliminate because it is not gradient-locked.
4. **In-place ASR Reconstruction (`burst_crit: 25.0`)**: High-variance transient bursts (jaw clenches, head motion spikes) are gently repaired in-place by Artifact Subspace Reconstruction (ASR) with `BurstRejection = 'off'`. No channels are removed and no time intervals are cut.
5. **Balanced ICLabel Rejection Rule**:
   * Class `Other` is **preserved** (not automatically deleted, protecting non-standard cortical dipoles).
   * Rejects only confirmed `Eye` (blinks/saccades, $\ge 0.70$), `Heart` (residual BCG, $\ge 0.70$), `Line` / `Channel` noise ($\ge 0.75$), and high-confidence `Muscle` ($P \ge 0.80$ with $P(\text{Brain}) < 0.10$).
   * **Over-cleaning Guard**: A strict ceiling guarantees that **no more than 35% of components** can ever be rejected.
6. **Automatic Provenance & Reproducibility (`pipeline_info.yaml`)**:
   Every cleaned folder (`data/<sid>/derivatives/05_ica/<seg>/`) automatically receives a `pipeline_info.yaml` with the exact Git branch, commit hash, data provenance (input/output paths), cleaning metrics, and full config snapshot.

---

## 🚀 Execution Guide: Common Scenarios & Examples

### 1️⃣ Сценарий 1: Полная обработка с нуля (From Scratch: 5000 Hz `.vhdr` / `.eeg`)

Если у вас есть только исходные записи в `EEG_ORIGINAL_ROOT`:

#### Вариант А: Рекомендуемый двухстадийный батч (P01 ➔ P02)
Этот вариант автоматически очищает тяжёлые промежуточные файлы 5000 Гц (~1 ГБ на сегмент) после ресемплинга:
```bash
# Шаг 1: Градиентный артефакт (Bergen AAS) + ресемплинг 5000 -> 250 Гц
python cleanP01.py --subjects 1916

# Шаг 2: BCG (OBS) + оптимизация ICA + финальная чистка (сохраняя каналы)
python cleanP02.py --subjects 1916

# Или запустить пакетно для всех испытуемых:
python cleanP01.py --all
python cleanP02.py --all
```

#### Вариант Б: Монолитный оркестратор `run_all.py` (Шаги 01–12)
```bash
# Обработать одного субъекта:
python run_all.py --subject 1916

# Обработать одного субъекта с принудительным пересчётом с нуля:
python run_all.py --subject 1916 --recalc

# Обработать один конкретный сегмент:
python run_all.py --segment-dir data/1916/segments/ec --recalc

# Обработать всех субъектов в data/:
python run_all.py --all
```

---

### 2️⃣ Сценарий 2: Запуск с шага 250 Гц (`data/<subject>/derivatives/02_resampled250`)

Если градиентный артефакт уже устранён и данные ресемплированы в 250 Гц (файлы `data/<subject>/derivatives/02_resampled250/<seg>/<seg>_250hz.fif`):

**Вам не нужно запускать Шаги 01–07!** Можно сразу стартовать с этапа P02:

```bash
# 1. Запустить P02 для одного субъекта (автоматически подхватит 02_resampled250):
python cleanP02.py --subjects 1916

# 2. Принудительный пересчёт всех шагов P02 (BCG + ICA):
python cleanP02.py --subjects 1916 --recalc
# (или эквивалентно: python cleanP02.py --subjects 1916 --force)

# 3. Запустить только для выбранных сегментов:
python cleanP02.py --subjects 1916 --segments ec eo

# 4. Запустить для всех испытуемых, у которых есть папка 02_resampled250:
python cleanP02.py --all

# 5. Проверить план выполнения без реального запуска (Dry-run):
python cleanP02.py --subjects 1916 --dry-run
```

#### Запуск отдельными скриптами по шагам:
```bash
# Шаг 08: Удаление BCG (fMRIB OBS). Автоматически читает 02_resampled250/<seg>_250hz.fif
python step08_bcg.py --subject 1916 --segment ec

# Шаг 10: Подбор параметров ICA на BCG-очищенных данных (03_bcg)
python step10_optuna_ica.py --subject 1916 --segment ec

# Шаг 11: Финальная ICA + ASR + ICLabel на всём сегменте (05_ica)
python step11_ica_final.py --subject 1916 --segment ec

# Шаг 12: Сводный HTML-отчёт
python step12_summary_report.py --subject 1916 --segment ec
```

---

### 3️⃣ Сценарий 3: Запуск с этапа BCG (`data/<subject>/derivatives/03_bcg`)

Если BCG уже посчитан (`<seg>_bcg_clean.fif`), но вы хотите пересчитать только ICA (например, с новыми щадящими настройками сохранения каналов):

```bash
# Пропустить шаг BCG и выполнить только ICA:
python cleanP02.py --subjects 1916 --set run.steps.bcg=false

# Принудительно пересчитать финальный шаг ICA (ica_final):
python cleanP02.py --subjects 1916 --force ica_final

# Запустить с переопределением параметров из командной строки:
python cleanP02.py --subjects 1916 --set ica_final.defaults.burst_crit=20.0
```

---

## ⚙️ Управление параметрами через `config/cleanP02.yaml`

Все параметры стадий BCG и ICA вынесены в [config/cleanP02.yaml](config/cleanP02.yaml).

| Параметр | Значение | Описание |
|---|:---:|---|
| `iclabel.ica_pca_max` | `35` | PCA-редукция до 35 компонент перед ICA: предотвращает расщепление компонент и переочистку |
| `filter.hp` / `filter.lp` | `1.0` / `45.0` | Полосовой фильтр (45.0 Гц полностью срезает вибрационную гребёнку крионасоса МРТ 44–100 Гц) |
| `bcg.npc_grid` | `[1, 2, 3]` | Сетка компонент fMRIB OBS ($\le 3$ эффективно убирает пульс и сохраняет альфа-ритм) |
| `ica_final.defaults.channel_crit` | `"off"` | Отключение удаления каналов по корреляции с соседями (все 95 каналов сохраняются) |
| `ica_final.defaults.burst_crit` | `25.0` | Порог ASR (SD cutoff): локально и мягко восстанавливает всплески движения во времени |
| `ica_final.defaults.iclabel_thresh` | `0.75` | Порог вероятности ICLabel для удаления артефактных компонент |
| `ica_opt.max_channel_removal_fraction` | `0.05` | Жёсткий лимит: выбрасывать не более 5% каналов (макс ~4–5) |
| `ica_final.guards.max_channel_crit` | `0.70` | Защита от перечистки: блокирует агрессивные значения порога удаления каналов |

Вы можете переопределять параметры прямо в командной строке:
```bash
python cleanP02.py --subjects 1916 --set bcg.fixed_npc=3 --set ica_final.defaults.burst_crit=25.0
```

---

## 📂 Структура директорий и производных (`derivatives`)

```
data/
└── <subject_id>/                       # e.g., 1916, 1924, 1929
    ├── segments/<seg>/                 # ec, eo, drone, lasertag, video
    │   ├── optuna_ica_best_params.json # Выигравшие параметры ICA (Шаг 10)
    │   └── optuna_ica_result.png       # График оптимизации Шага 10
    │
    └── derivatives/
        ├── 01_bergen/<seg>/            # Шаг 05: Bergen AAS 5000 Hz (.set)
        ├── 01_bergen_250hz/<seg>/      # P01: Bergen AAS 250 Hz (.fif)
        ├── 02_resampled250/<seg>/      # Ресемплированные данные 250 Hz (<seg>_250hz.fif)
        ├── 03_bcg/<seg>/               # Шаг 08: BCG-очищенные данные (<seg>_bcg_clean.fif)
        ├── 04_channels/<seg>/          # Шаг 11: removed_channels.json (список удалённых каналов)
        └── 05_ica/<seg>/               # Шаг 11: Финальные данные:
            ├── <seg>_ica_clean.fif     #   - MNE чистые данные
            ├── <seg>_ica_clean.set     #   - EEGLAB чистые данные
            ├── <seg>_ica_report.html   #   - Интерактивный HTML-отчёт
            ├── <seg>_ica_metrics.json  #   - Числовые метрики очистки
            └── pipeline_info.yaml      #   - Паспорт воспроизводимости (Git коммит, ветка, пути, конфиг)
```

---

## 📖 Подробное описание каждого шага (Step 01 – Step 12)

### [STEP 01] Детекция МРТ-сессий (`step01_detect_mri.py`)
- **Цель**: Автоматический поиск границ включения градиентов МРТ по RMS-огибающей непрерывной 5000 Гц записи.
- Фильтрует короткие калибровочные импульсы (`MIN_NAMED_SEC = 250.0s`).
- Сортирует файлы движения SPM (`rp_*.txt`) и сопоставляет их с сегментами: `ec -> eo -> drone -> lasertag -> video`.

### [STEP 02] Субсэмпловая фаза срезов (`step02_detect_slices.py`)
- **Цель**: Кросс-корреляция профиля импульсов с гребёнчатым шаблоном периода среза (100 мс = 500 отсчётов при 5000 Гц).
- Выход: `slice_detection.json`.

### [STEP 03] Обрезка холостых сканов (Dummy Scans) (`step03_trim_dummy.py`)
- **Цель**: Синхронизация начала ЭЭГ с началом реальных функциональных объёмов fMRI и отсечение переходных сканов.
- Выход: `slice_triggers.txt`, `segment_work_info.json`.

### [STEP 04] Оптимизация параметров Bergen AAS (`step04_optuna_tune.py`)
- **Цель**: Байесовская оптимизация параметров фазового сдвига (`shift`), окна усреднения (`win_k`) и порога движения (`motion_thresh`) на затылочном канале `Oz`.

### [STEP 05] Полноканальная чистка Bergen AAS (`step05_bergen_clean.py`)
- **Цель**: Подавление градиентного артефакта на всех 96 каналах в MATLAB с экономией памяти (`float32`).

### [STEP 06] Спектральный анализ альфа-ритма (`step06_spectra_analysis.py`)
- **Цель**: Оценка подавления гармоник градиента (20, 30, 40, 50, 60 Гц) и сохранности альфа-ритма (8–13 Гц).

### [STEP 07] HTML-отчёт Bergen (`step07_html_report.py`)
- **Цель**: Генерация интерактивного отчёта о качестве удаления градиентного артефакта.

### [STEP 08] Удаление баллистокардиограммы (BCG) (`step08_bcg.py`)
- **Цель**: 
  1. Чтение 250 Гц данных (из `02_resampled250` или даунсэмплинг из 5000 Гц `.set`).
  2. Полосовой фильтр 1.0–45.0 Гц (1.0 Гц полностью убирает дыхательный дрейф; 45.0 Гц срезает гармоническую гребенку крионасоса томографа).
  3. Детекция QRS-пиков кардиоканала в fMRIB (`fmrib_pas`).
  4. Подбор оптимального числа компонент базиса (`npc = 1..3`) с контролем сохранения альфа-ритма.
- Выход: `derivatives/03_bcg/<seg>/<seg>_bcg_clean.fif`.

### [STEP 10] Быстрая двухфазная оптимизация ICA (`step10_optuna_ica.py`)
- **Цель**: 
  - **Фаза 1**: Сетка `clean_rawdata` с приоритетом сохранения каналов (`channel_grid: ["off"]`).
  - **Фаза 2**: Однократный запуск Picard ICA с PCA-редукцией до 35 компонент (`ica_pca_max = 35`).
  - **Фаза 3**: Быстрый подбор порога ICLabel (0.70–0.85) по сбалансированным правилам (сохранение класса Other, удаление подтверждённых артефактов).
- Выход: `optuna_ica_best_params.json`.

### [STEP 11] Финальная чистка ICA + ASR (`step11_ica_final.py`)
- **Цель**: 
  1. Применение ASR (`BurstCriterion = 25.0`) для мягкой реконструкции всплесков во времени без вырезания каналов.
  2. Picard ICA с редукцией до 35 компонент на всем сегменте.
  3. Сбалансированная классификация компонент через `mne-icalabel`: удаление артефактов (Eye, Heart, high-confidence Muscle, Line/Chan), сохранение класса `Other`, и защитный потолок (не более 35% компонент).
  4. Сохранение всех каналов в исходном виде.
  5. Перевод к общему усреднённому референсу (Average Reference).
  6. Экспорт паспорта воспроизводимости `pipeline_info.yaml` (Git коммит, ветка, пути входа/выхода, конфиг).
- Выход: `derivatives/05_ica/<seg>/<seg>_ica_clean.fif`, `.set`, и `pipeline_info.yaml`.

### [STEP 12] Сводный многостадийный отчёт (`step12_summary_report.py`)
- **Цель**: Построение интерактивного дашборда сравнения **Сырая ЭЭГ ➔ Bergen AAS ➔ BCG OBS ➔ Финальная ICA**:
  - Наложения спектров PSD до и после каждой стадии.
  - Метрики сохранности физиологического сигнала и подавления артефактов.

---

## 📊 Таблица целевых критериев качества (Quality Targets)

| Метрика | Целевое значение | Предупреждение / Флаг REVIEW |
|---|:---:|:---:|
| **Подавление градиента fMRI (20–60 Гц)** | $\ge 99.5\%$ | $< 99.0\%$ |
| **Сохранение альфа-ритма после Bergen** | $\ge 85.0\%$ | $< 70.0\%$ |
| **Подавление кардио-артефакта BCG (0.7–4.0 Гц)** | $\ge 25.0\%$ | $< 10.0\%$ |
| **Сохранение альфа-ритма после BCG** | $\ge 60.0\%$ | $< 50.0\%$ |
| **Доля удалённых IC компонент** | $20\% - 40\%$ | $> 50\%$ |
| **Падение общей дисперсии после ICA** | $\le 40.0\%$ | $> 60\%$ |
| **Число удалённых каналов** | **0** (все сохранены) | $> 5$ каналов |
