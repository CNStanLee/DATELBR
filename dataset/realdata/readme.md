03: 7kw 18A
04: 7kw 18A
05: 7kw 18A
06: 6kw 15A
07: 6kw 15A
08: 6kw 15A
09: 4.8kw 12A
010: 4.8kw 12A
011: 4.8kw 12A
012: 3.6kw 9A
13
14
15: 2.4kw 6A
16
17
18: 1.2kw 3A
19
20
21: v2g 2.3kw 6A
22: BACKGROUND NOISE

23: battery 500w
24: battery 1000W
25: 1500
26: 2000
27: 2500
28: 3000
29: 100
30: 300
31: 100 discharging to grid
32: 300
33: 500
34: 1000

## THDi脚本使用说明

脚本位置：`datacheck/data_inspect.py`

### 1) 使用py310环境运行（默认分析3A.csv和V2G_6A.csv）

```bash
cd /home/changhong/Documents/Waveforms/datacheck
/home/changhong/anaconda3/envs/py310/bin/python data_inspect.py
```

### 2) 分析datacheck目录下所有CSV

```bash
cd /home/changhong/Documents/Waveforms/datacheck
/home/changhong/anaconda3/envs/py310/bin/python data_inspect.py --all
```

### 3) 指定要分析的CSV文件

```bash
cd /home/changhong/Documents/Waveforms/datacheck
/home/changhong/anaconda3/envs/py310/bin/python data_inspect.py --files 3A.csv 6A.csv 9A.csv 12A.csv 15A.csv 18A.csv V2G_6A.csv
```

### 4) 指定汇总输出文件名

```bash
cd /home/changhong/Documents/Waveforms/datacheck
/home/changhong/anaconda3/envs/py310/bin/python data_inspect.py --all --out thdi_summary_all.csv
```

运行后会输出：
- 每个CSV对应的检查图：`*_inspect.png`
- THDi汇总表：默认 `thdi_summary.csv`（可用 `--out` 修改）