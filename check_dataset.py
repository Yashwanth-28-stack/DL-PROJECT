import os

dataset_path = "dataset"

train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

print("Checking dataset...\n")

print("Train folder exists:", os.path.exists(train_path))
print("Test folder exists:", os.path.exists(test_path))

print("\nTraining classes:")
for folder in os.listdir(train_path):
    print(folder)

print("\nTesting classes:")
for folder in os.listdir(test_path):
    print(folder)