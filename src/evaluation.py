import torch
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(
    model,
    data_loader,
    fabric_classes,
    defect_classes,
    device
):
    model.eval()

    correct_fabric = 0
    correct_defect = 0
    total = 0

    all_fabric_true = []
    all_fabric_pred = []

    all_defect_true = []
    all_defect_pred = []

    with torch.no_grad():

        for images, fabric_labels, defect_labels in data_loader:

            images = images.to(device)
            fabric_labels = fabric_labels.to(device)
            defect_labels = defect_labels.to(device)

            fabric_output, defect_output = model(images)

            fabric_predictions = fabric_output.argmax(dim=1)
            defect_predictions = defect_output.argmax(dim=1)

            correct_fabric += (fabric_predictions == fabric_labels).sum().item()

            correct_defect += (defect_predictions == defect_labels).sum().item()

            total += images.size(0)

            all_fabric_true.extend(fabric_labels.cpu().tolist())

            all_fabric_pred.extend(fabric_predictions.cpu().tolist())

            all_defect_true.extend(defect_labels.cpu().tolist())

            all_defect_pred.extend(defect_predictions.cpu().tolist())

    fabric_accuracy = correct_fabric / total
    defect_accuracy = correct_defect / total
    average_accuracy = (fabric_accuracy + defect_accuracy) / 2

    fabric_report = classification_report(
        all_fabric_true,
        all_fabric_pred,
        labels=list(range(len(fabric_classes))),
        target_names=fabric_classes,
        zero_division=0
    )

    defect_report = classification_report(
        all_defect_true,
        all_defect_pred,
        labels=list(range(len(defect_classes))),
        target_names=defect_classes,
        zero_division=0
    )

    defect_matrix = confusion_matrix(
        all_defect_true,
        all_defect_pred,
        labels=list(range(len(defect_classes)))
    )

    return {
        "fabric_accuracy": fabric_accuracy,
        "defect_accuracy": defect_accuracy,
        "average_accuracy": average_accuracy,
        "fabric_report": fabric_report,
        "defect_report": defect_report,
        "defect_matrix": defect_matrix,
    }
