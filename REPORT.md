# Короткий отчёт: World Model + VLM scorer

## Setup

- Среда: `TinyGridGoalEnv` (дискретные действия, RGB-кадры).
- World model: компактный RSSM (в стиле Dreamer/PlaNet):
  - encoder `obs -> embedding`
  - recurrent latent dynamics (GRU + stochastic state)
  - decoder и головы `reward/done`
- Планирование: random shooting MPC на горизонте `H=10`, выбирается лучший action-sequence, выполняется только первый action.
- VLM scorer: предобученный CLIP (`openai/clip-vit-base-patch32`) для текстовой цели `"agent at the green goal"`, скоринг применяется к **imagined future frames**.

## Сравнение с baseline

Сравниваются 3 режима:
1. `random`
2. `wm_no_vlm` (objective = predicted return в world model)
3. `wm_vlm` (objective = predicted return + VLM score)

Метрики сохраняются в `outputs/metrics.json` после запуска:

```bash
python run_demo.py --device cpu --train-episodes 120 --train-epochs 12 --eval-episodes 12
```

В текущем sandbox-окружении зависимости для обучения (PyTorch/Transformers) недоступны, поэтому фактический прогон нужно выполнить в стандартном Python-окружении с доступом к пакетам из `requirements.txt`.

## Визуализация

Скрипт автоматически сохраняет GIF первого эпизода каждого метода:
- `outputs/random_ep0.gif`
- `outputs/wm_no_vlm_ep0.gif`
- `outputs/wm_vlm_ep0.gif`

## Failure modes

Наблюдаемые/ожидаемые проблемы:
- Ошибки model-bias: RSSM может переоценивать плохие траектории на длинном горизонте.
- Расхождение между реконструкцией decoder и реальными кадрами снижает качество VLM-скора.
- CLIP-чувствительность к артефактам синтетических кадров, если world model генерирует «размытые» состояния.
- Random shooting неэффективен при большом action space и длинном горизонте.

## Future work

- Заменить random shooting на CEM.
- Добавить uncertainty penalty (ensemble world models).
- Использовать stronger decoder / latent overshooting.
- Перейти на MiniGrid и сравнить несколько текстовых целей.
- Добавить actor-critic (Dreamer-style policy learning в latent space).
