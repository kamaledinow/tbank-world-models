# World Model + VLM Scorer Demo

Демо-проект для задания: планирование действий агента через imagined rollouts в RSSM world model и оценка воображаемых будущих кадров с помощью VLM (CLIP).

## Что реализовано

- Простая среда `TinyGridGoalEnv` (grid-world с RGB-наблюдением, цель — дойти до зелёной клетки).
- Компактная RSSM world model (encoder + recurrent latent dynamics + decoder/reward/done heads).
- Планирование random shooting в learned world model на горизонте `H`.
- VLM-based scorer: CLIP (`openai/clip-vit-base-patch32`) оценивает воображаемые кадры относительно текстовой цели (например, `agent at the green goal`).
- Baselines:
  - `random`
  - `wm_no_vlm` (planning по reward world model без VLM-скора)
  - `wm_vlm` (planning по reward + VLM score imagined futures)

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python run_demo.py --device cpu --train-episodes 120 --train-epochs 12 --eval-episodes 12
```

Артефакты появятся в `outputs/`:
- `metrics.json`
- `random_ep0.gif`
- `wm_no_vlm_ep0.gif`
- `wm_vlm_ep0.gif`

## Краткий отчёт

См. `REPORT.md` (содержит количественные результаты, обсуждение failure modes и future work).
