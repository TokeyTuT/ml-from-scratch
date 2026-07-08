# Datasets

本目录存放用于本地算法练习的小型样例数据。

## california_housing_sample.csv

该文件是从 `sklearn.datasets.fetch_california_housing` 中抽样得到的 500 行数据，生成时使用 `random_state=42`。

字段说明：

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`
- `MedHouseVal`

回归练习中，前 8 列可作为特征，`MedHouseVal` 可作为目标值。
