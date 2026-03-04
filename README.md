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


## Troubleshooting

### Ошибка `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

Это конфликт бинарной совместимости между установленными версиями `numpy` и `torch`.

Быстрое решение (рекомендуется в новом venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip uninstall -y numpy torch torchvision torchaudio
pip install "numpy<2"
pip install "torch>=2.2,<2.4"
pip install "transformers>=4.40,<4.46"
pip install -r requirements.txt
```

Проверка, что всё в порядке:

```bash
python -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__); import numpy as np; import torch as t; t.from_numpy(np.zeros(1, dtype=np.float32)); print('OK')"
```

Если у вас уже установлен `torch==2.2.2`, лучше обновить его до `2.4+` **или** оставить 2.2.2 и строго зафиксировать `numpy<2`.


### Ошибка `Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2`

Это означает, что установлен слишком новый `transformers`, который отключает backend PyTorch<2.4.

Исправление (для `torch==2.2.x`):

```bash
pip uninstall -y transformers
pip install "transformers>=4.40,<4.46"
```

Либо альтернативно обновить torch до `>=2.4` и оставить более новую версию `transformers`.


### Скрипт "висит" после скачивания `pytorch_model.bin`

Обычно это не зависание, а очень медленный CPU-инференс CLIP во время `wm_vlm` планирования.

Почему долго:
- на каждом шаге planner оценивает много candidate-траекторий;
- для каждой траектории раньше считался CLIP-скор отдельно, что на CPU может занимать десятки минут/часы.

Что делать:
1. Обновиться до последней версии этого репозитория (там добавлен batched VLM scoring).
2. Запустить с более лёгкими параметрами:

```bash
python run_demo.py --device cpu --train-episodes 60 --train-epochs 6 --eval-episodes 4 --horizon 6 --num-candidates 6
```

3. Для полноценного прогона лучше использовать GPU (`--device cuda`, если доступно).

Если нужно быстро проверить только обучение world model без VLM-части, временно отключите policy `wm_vlm` в цикле evaluation в `run_demo.py`.


### Ошибка `run_demo.py: error: unrecognized arguments: --horizon ... --num-candidates ...`

Скорее всего запускается **не та версия** `run_demo.py` (старый файл/папка) или не тот интерпретатор.

Проверьте:

```bash
pwd
python -c "import sys; print(sys.executable)"
python run_demo.py -h
```

В `-h` должны быть аргументы `--horizon` и `--num-candidates`.

Если их нет:
1. Обновите репозиторий (`git pull`) или заново скачайте архив с последними изменениями.
2. Запускайте из корня проекта, где лежит актуальный `run_demo.py`.
3. Дополнительно поддерживаются алиасы: `--planning-horizon` и `--num_candidates`.
<<<<<<< codex/create-demo-project-for-world-model-and-vlm-20y1fn


### Ошибка `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`

Это бывает из-за различий в API/обёртках `transformers` в разных окружениях: вместо тензора иногда возвращается объект model output.

В актуальной версии репозитория `CLIPScorer` уже обрабатывает оба варианта (тензор и model output).

Что сделать:
1. Обновить репозиторий до последнего коммита.
2. Переустановить зависимости в venv:

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

3. Перезапустить запуск.

Сообщение про `HF_TOKEN` — это предупреждение про лимиты Hub, не фатальная ошибка.
=======
>>>>>>> main
