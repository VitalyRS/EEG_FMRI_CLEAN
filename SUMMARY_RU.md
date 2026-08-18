# 📋 Сводка: Оптимизация ICA через Optuna

⚠️ **ВАЖНО:** Прочти [QUALITY_CRITERIA.md](QUALITY_CRITERIA.md) перед запуском — там описаны целевые метрики, типичные ловушки переочистки и как их диагностировать.

## Целевые метрики качества

### ✅ После Bergen:
- Подавление градиентов (20/30/40/50/60 Hz): **≥ 99.5%**
- Сохранение альфы: **≥ 85%**

### ✅ После BCG:
- Подавление кардио (0.7–4 Hz): **≥ 20%** (меньше — допустимо, если BCG слабый)
- Сохранение альфы: **≥ 60%**

### ✅ После ICA:
- Отклонено компонент: **20–40%** (НЕ 60–80%!)
- Variance drop: **≤ 30%**
- Сохранение альфы: **≥ 70%**
- Удалено каналов: **≤ 10%**

### 🚩 Красные флаги (STOP и пересмотри параметры):
- Отклонено > 60% компонент → порог ICLabel слишком низкий
- Variance drop > 50% → переочистка, удалили реальный сигнал
- Альфа < 60% → параметры слишком агрессивны
- Удалены Fz, Cz, CPz → ChannelCriterion слишком строгий

---

Добавлены 4 новых шага пайплайна (08-11) для автоматической очистки остаточных артефактов (BCG, мышцы, моргания) после Bergen с полной Байесовской оптимизацией параметров:

### Step 08: `step08_bcg_optuna.py`
- Оптимизирует параметры удаления BCG (баллистокардиограмма) через Optuna TPE
- Параметры поиска:
  - `pre_filt`: полосовой фильтр перед детекцией пиков (0.5-40 Hz)
  - `l_freq`, `h_freq`: финальный фильтр после удаления
  - `corr_thresh`: порог корреляции для шаблона (0.5-0.95)
- Функция потерь: баланс между подавлением BCG (↓ мощность 1-5 Hz) и сохранением альфа (↑ 8-13 Hz)
- Выход: `bcg_optuna_best.json`, `bcg_optuna_study.db`, `bcg_optuna_result.png`

### Step 09: `step09_ica.py`
- Применяет найденные оптимальные параметры BCG к полным данным
- Даунсэмплинг Bergen .set → 250 Hz MNE .fif
- Удаление BCG через `mne.preprocessing.find_ecg_events`
- Выход: `data/.../03_bcg/segment4_bcg_clean.fif`

### Step 10: `step10_optuna_ica.py`
- Оптимизирует **все** параметры `clean_rawdata` + порог ICLabel
- Параметры поиска (6 измерений):
  - `flatline_crit`: 3-8 сек (обнаружение мёртвых каналов)
  - `channel_crit`: 0.6-0.95 (корреляция с соседями)
  - `line_crit`: 2-6 (50 Hz линейный шум)
  - `burst_crit`: 15-40 (ASR отключён, но параметр сохранён для совместимости)
  - `window_crit`: 0.2-0.6 (скользящее окно)
  - `iclabel_thresh`: 0.60-0.90 (порог удаления артефактных компонент)
- Полный ICA на **первых 60 секундах** (для скорости: runica медленный)
- Функция потерь:
  ```
  Loss = 100×(0.85 - alpha_retention)² + 50×max(0, variance_drop - 0.15)² + 200×(n_ch_removed/95)²
  ```
- Выход: `ica_optuna_best.json`, `ica_optuna_study.db`, `ica_optuna_result.png`

### Step 11: `step11_ica_final.py`
- Применяет найденные оптимальные параметры к **полным данным**
- Интерполяция плохих каналов → реф к average → ICA (Extended Infomax, 25 компонент) → ICLabel → удаление артефактов
- Выход: `data/.../05_ica/segment4_ica_clean.fif` + HTML-отчёт с топографией компонент

## Обновлён `run_all.py`

Добавлены флаги:
- `--skip-bcg`: пропустить шаги 08-09 (BCG)
- `--skip-ica-optuna`: пропустить шаг 10 (использовать существующие параметры ICA)
- `--skip-ica`: пропустить шаги 10-11 полностью

## Как запустить

### Полный пайплайн (Bergen → BCG → ICA):
```bash
python run_all.py
```

### Только Bergen (как раньше):
```bash
python run_all.py --skip-bcg --skip-ica
```

### Только ICA-часть (Bergen уже выполнен):
```bash
python step08_bcg_optuna.py      # ~5-10 мин
python step09_ica.py             # ~1-2 мин
python step10_optuna_ica.py      # ~40-60 мин (20 триалов × 2-3 мин на ICA)
python step11_ica_final.py       # ~3-5 мин
```

## Время выполнения

- **Step 08 (BCG Optuna)**: ~5-10 мин (20 триалов, без ICA)
- **Step 09 (BCG apply)**: ~1-2 мин
- **Step 10 (ICA Optuna)**: ~40-60 мин (20 триалов, каждый с полным ICA на 60 сек)
- **Step 11 (ICA apply)**: ~3-5 мин

**Итого**: ~50-80 минут для BCG+ICA с дефолтными 20 триалами.

## Зависимости

Нужен `mne-icalabel`:
```bash
conda activate NLP_ENV
pip install mne-icalabel
```

**MATLAB НЕ НУЖЕН для шагов 08-11** — это чистый Python + MNE.

## Что проверить перед запуском

1. **Удали старые `.db` файлы оптимизации** (если хочешь чистый прогон):
   ```bash
   rm 1916/segments/segment4/bcg_optuna_study.db
   rm 1916/segments/segment4/ica_optuna_study.db
   ```

2. **Проверь, что Bergen шаги 01-05 завершены**:
   - Должен существовать: `1916/segments/segment4/segment4_bergen_optuna_*.set`

3. **Первый запуск:** используй дефолтные 20 триалов (уже установлено в `config.py`)

## Примеры лучших параметров

Типичные значения для subject 1916, segment4 (могут отличаться для твоих данных):

**BCG**:
```json
{
  "pre_filt": [0.5, 5.0],
  "l_freq": 0.5,
  "h_freq": 40.0,
  "corr_thresh": 0.75
}
```

**ICA**:
```json
{
  "flatline_crit": 5,
  "channel_crit": 0.8,
  "line_crit": 4,
  "burst_crit": 25,
  "window_crit": 0.3,
  "iclabel_thresh": 0.75
}
```

## Выходные файлы

```
1916/segments/segment4/
├── bcg_optuna_best.json          # Победители BCG
├── bcg_optuna_study.db           # История триалов BCG
├── bcg_optuna_result.png         # График оптимизации BCG
├── ica_optuna_best.json          # Победители ICA
├── ica_optuna_study.db           # История триалов ICA
└── ica_optuna_result.png         # График оптимизации ICA

data/1916/derivatives/
├── 03_bcg/segment4/
│   └── segment4_bcg_clean.fif    # После BCG
└── 05_ica/segment4/
    ├── segment4_ica_clean.fif    # Финальные чистые данные
    └── segment4_ica_report.html  # Отчёт с топографией компонент
```

## Важные замечания

1. **Step 10 использует только первые 60 секунд** для ускорения (ICA медленный). Найденные параметры применяются к полным данным в step 11.

2. **ICLabel порог**:
   - Высокий (0.85-0.90) → агрессивная очистка, больше variance_drop
   - Низкий (0.60-0.70) → консервативная, сохраняет больше сигнала

3. **ASR отключён** — оптимизируем только детекцию плохих каналов и линейный шум.

4. **Все промежуточные .fif файлы** хранятся в `data/1916/derivatives/` по BIDS-подобной структуре.

## Следующий шаг

Запусти:
```bash
cd /home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN
python run_all.py --skip-detect-mri
```

Это выполнит:
- Шаги 02-07: Bergen (с исправленным TR_sl — без баг с 20/30/40 Hz)
- Шаги 08-09: BCG removal
- Шаги 10-11: ICA с оптимизацией

После завершения пришли мне:
1. `segment4_cleaning_report.html` (Bergen)
2. `segment4_ica_report.html` (ICA)
3. `ica_optuna_best.json` (найденные параметры)

Я проверю результаты!

## ⚠️ Рекомендованные параметры ICA

**КОНСЕРВАТИВНЫЕ** (для предотвращения переочистки):
```json
{
  "flatline_crit": 5.0,
  "channel_crit": 0.75,        // НЕ ВЫШЕ 0.80!
  "line_crit": 4.0,
  "iclabel_thresh": 0.80       // НЕ НИЖЕ 0.75!
}
```

### Если в результатах видишь:
- **Отклонено 60–80% компонент** → `iclabel_thresh: 0.85`
- **Удалены Fz, Cz, CPz** → `channel_crit: 0.70` или `'off'`
- **Variance drop > 50%** → оба параметра слишком агрессивны

### ⚠️ Типичная ошибка:
`iclabel_thresh: 0.60` → удаляет компоненты с P(Brain)=0.55 (**вероятно мозг!**)
→ Результат: 70% компонент отклонено, variance drop 70%, альфа съедена
→ **Минимум 0.75, лучше 0.80**

---

## 🔧 Как исправить переочистку

Если уже запустил и получил плохой результат:

1. **Удали старые параметры:**
```bash
rm data/1916/segments/segment04/ica_optuna_best.json
rm data/1916/segments/segment04/optuna_ica_best_params.json
```

2. **Отредактируй дефолты в step11_ica_final.py** (строка ~62):
```python
best = {
    "flatline_crit": 5.0,
    "channel_crit": 0.75,      # было 0.80
    "line_crit": 4.0,
    "iclabel_thresh": 0.80     # было 0.70
}
```

3. **Перезапусти:**
```bash
python step11_ica_final.py --segment-dir data/1916/segments/segment04
```

4. **Проверь метрики:**
```bash
cat data/1916/derivatives/05_ica/segment04/segment04_ica_metrics.json
```

Целевые значения:
- `n_ic_rejected` / `n_ic`: 0.20–0.40 (не больше!)
- `variance_drop`: ≤ 0.30
- `alpha_retention`: ≥ 0.70

---

## 📚 Дополнительная документация

- **[QUALITY_CRITERIA.md](QUALITY_CRITERIA.md)** — полный гид по критериям качества и диагностике
- **[QUICKSTART_ICA.md](QUICKSTART_ICA.md)** — быстрый старт для ICA-части (на английском)
- **[README.md](README.md)** — полная документация пайплайна
