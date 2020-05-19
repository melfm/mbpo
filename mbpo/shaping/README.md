1. Train an expert
```
python mbpo/shaping/main.py --train_behavioral
```

2. Rollout Demo Data
```
python mbpo/shaping/main.py --generate_buffer
```

3. Train the potential function
```
python mbpo/shaping/train_shaping.py
```

4. Train MBPO with reward_shaping enabled
```
mbpo run_local examples.development --config=examples.config.halfcheetah.0 --gpus=1 --trial-gpus=1
```