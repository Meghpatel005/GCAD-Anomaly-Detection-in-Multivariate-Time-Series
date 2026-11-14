# Dataset layout

The benchmark datasets are not redistributable and are not included. Download them from their original
sources and arrange them under `data/` (the `data.root` config value) exactly as below. Loaders fail with an
explicit "missing file" error if something is not where it is expected. Statistics from the paper's Table 1 are
in `src/gcad/paper_results.py::TABLE1`.

| Dataset | Channels | Train | Test | Anomalies |
|---|---|---|---|---|
| SWaT | 51 | 47,520 | 44,991 | 12.20% |
| SMD | 38 | 28,479 | 28,479 | 9.46% |
| MSL | 55 | 3,682 | 2,856 | 0.74% |
| SMAP | 25 | 2,876 | 8,579 | 2.14% |
| PSM | 25 | 132,481 | 87,841 | 27.76% |

## SWaT (iTrust, SUTD – request access from the provider)

```
data/SWaT/SWaT_Dataset_Normal_v1.csv   # normal operation, CSV export of the original xlsx
data/SWaT/SWaT_Dataset_Attack_v0.csv   # attack period, has the "Normal/Attack" label column
```
File names can be overridden with `data.train_file` / `data.test_file`. The timestamp column is dropped.
The paper's test length (44,991) matches 10x block-mean downsampling (`data.downsample: 10`). Its training length
(47,520) matches 49,680 downsampled rows minus the first 2,160, the usual removal of the plant's start-up period,
so `configs/swat.yaml` sets `data.skip_head: 21600` (raw rows). Both are inferred from the published numbers; the
loader logs a warning if the resulting sizes differ from Table 1.

## SMD (OmniAnomaly, `ServerMachineDataset`)

```
data/SMD/train/machine-1-1.txt
data/SMD/test/machine-1-1.txt
data/SMD/test_label/machine-1-1.txt
```
Select the machine with `data.entity` (default `machine-1-1`, 38 channels).

## MSL and SMAP (NASA telemanom)

```
data/NASA/train/<chan_id>.npy
data/NASA/test/<chan_id>.npy
data/NASA/labeled_anomalies.csv      # columns: chan_id, spacecraft, anomaly_sequences, ...
```
`data.entity` is a `chan_id` of the chosen spacecraft (MSL: 55 channels, SMAP: 25). One entity is used per run;
entities are not concatenated because sliding windows would straddle channel boundaries. The paper's Table 1
sizes for MSL and SMAP describe a single unnamed entity. Anomaly ranges come from `anomaly_sequences`
(inclusive indices).

## PSM (eBay)

```
data/PSM/train.csv
data/PSM/test.csv
data/PSM/test_label.csv              # column "label"
```

## Your own data

Create `data/dataset.npz` with arrays `train` (T×N), `test` (T×N) and `labels` (T,) and set `data.name: npz`.

## Preprocessing (all datasets)

`data.skip_head` raw training rows are dropped first, then `data.downsample` is applied. The last
`data.val_fraction` (20%) of the normal data is held out chronologically for validation; channels are
standardised with statistics of the training part only (constant channels get std = 1); missing values are
forward/back-filled.
