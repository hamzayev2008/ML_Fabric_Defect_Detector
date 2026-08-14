from dataLoader import (
    train_dataset,
    validation_dataset,
    test_dataset
)


print("TRAIN")
print("Fabric classes:")
print(train_dataset.fabric_classes)

print("\nDefect classes:")
print(train_dataset.defect_classes)

print("\nNumber of train images:")
print(len(train_dataset))


print("\nVALIDATION")
print("Number of validation images:")
print(len(validation_dataset))


print("\nTEST")
print("Number of test images:")
print(len(test_dataset))