# 🚑 Шпаргалка: Быстрая диагностика качества очистки

## 📊 Проверь метрики за 30 секунд

```bash
cd /home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN

# Bergen
cat 1916/segments/segment04/summary_alpha_quality.csv | column -t -s,

# ICA
cat data/1916/derivatives/05_ica/segment04/segment04_ica_metrics.json | jq '{
  ic_rejected: "\(.n_ic_rejected)/\(.n_ic)",
  reject_pct: ((.n_ic_rejected / .n_ic * 100) | round),
  variance_drop: (.variance_drop * 100 | round),
  alpha_retention: (.alpha_retention * 100 | round),
  channels_removed: .n_channels_removed,
  threshold: .iclabel_thresh
}'
```

## ✅ Хорошие значения (копируй и сравнивай)

```
Bergen:    G_supp ≥99.5%  alpha≥85%
BCG:       cardiac≥20%    alpha≥60%
ICA:       reject≤40%     var_drop≤30%  alpha≥70%  ch_removed≤10
```

## 🚩 Красные флаги

| Симптом | Причина | Решение |
|---------|---------|---------|
| ICA reject > 60% | `iclabel_thresh` слишком низкий (≤0.65) | Подними до 0.80 |
| variance_drop > 50% | Переочистка (удалено слишком много) | iclabel_thresh → 0.85 |
| alpha < 60% | Параметры слишком агрессивны | Проверь каждую стадию |
| Удалены Fz/Cz/CPz | `channel_crit` > 0.85 | Снизь до 0.75 |
| Рост PSD после 50 Hz | Переочистка + алиасинг | Меньше отклонённых компонент |

## 🔧 Быстрые фиксы

### Fix 1: ICA отклонила 70% компонент
```bash
rm data/1916/segments/segment04/ica_optuna_best.json
# Отредактируй step11_ica_final.py строка 62: iclabel_thresh = 0.80
python step11_ica_final.py --segment-dir data/1916/segments/segment04
```

### Fix 2: Удалены центральные каналы (Fz, Cz, CPz)
Создай `data/1916/segments/segment04/ica_optuna_best.json`:
```json
{
  "best_params": {
    "flatline_crit": 5.0,
    "channel_crit": 0.70,
    "line_crit": 4.0,
    "iclabel_thresh": 0.80
  }
}
```
Перезапусти step11.

### Fix 3: BCG слабое подавление (< 20%)
Это **норма**, если Bergen уже подавил пульс. Не увеличивай агрессивность — съешь альфу.

## 📈 Целевая динамика метрик

```
После Bergen:  alpha ~ 85–95%
После BCG:     alpha ~ 60–75%  (падение допустимо)
После ICA:     alpha ~ 70–85%  (восстановление)
```

Если альфа монотонно падает (95→75→50) → каждая стадия слишком агрессивна.

## 🎯 Однострочная проверка

```bash
python3 -c "
import json
m = json.load(open('data/1916/derivatives/05_ica/segment04/segment04_ica_metrics.json'))
rej_pct = m['n_ic_rejected'] / m['n_ic'] * 100
var = m['variance_drop'] * 100
alp = m['alpha_retention'] * 100
status = 'GOOD' if rej_pct < 40 and var < 30 and alp > 70 else 'BAD'
print(f'{status}: reject {rej_pct:.0f}%, var_drop {var:.0f}%, alpha {alp:.0f}%')
if status == 'BAD':
    if rej_pct > 60: print('  → iclabel_thresh слишком низкий, подними до 0.80')
    if var > 50: print('  → переочистка, используй порог 0.85')
    if m['n_channels_removed'] > 9: print('  → channel_crit слишком строгий, снизь до 0.70')
"
```

## 📚 Полная документация

- **Типичные ловушки:** [QUALITY_CRITERIA.md](QUALITY_CRITERIA.md)
- **Инструкции на русском:** [SUMMARY_RU.md](SUMMARY_RU.md)
- **Полный README:** [README.md](README.md)
