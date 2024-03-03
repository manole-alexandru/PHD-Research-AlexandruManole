from data.datasets.RailSem19Dataset import RailSem19Dataset


print("Hello AI world")

# 0. OnlineDataset
# 1. Dataset
# 2. (Processing Components)
# 3. Dataset (in desired format)
# 4. Dataloader
# 5. Model
# 6. Training procedure
# 7. Evaludating procedure
# 8. Save Data in various ways

dataset = RailSem19Dataset('D:/Mano/Rail/RailSem19/')
print(dataset.__len__())
