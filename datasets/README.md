# Datasets

Small sample datasets for local algorithm practice.

## california_housing_sample.csv

Sample of 500 rows from `sklearn.datasets.fetch_california_housing`, generated
with `random_state=42`.

Columns:

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`
- `MedHouseVal`

For regression practice, use the first 8 columns as features and `MedHouseVal`
as the target.
