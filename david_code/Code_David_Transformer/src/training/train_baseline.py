"""
Baseline training: Supervised learning on source domain only.
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from ..utils.lr_schedule import schedule_dict


def test(model, data_loader, iter_num, device):
    """Evaluate model on test set."""
    model.eval()
    correct_class = 0
    
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            features, output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct_class += pred.eq(label.data.view_as(pred)).cpu().sum()
    
    accuracy = 100.0 * correct_class / len(data_loader.dataset)
    logging.info(f"Test iteration {iter_num}: Accuracy = {accuracy:.2f}%")
    
    return {
        "iter": iter_num,
        "correct_class": correct_class,
        "total_elems": len(data_loader.dataset),
        "accuracy %": accuracy
    }


def train_baseline(config, base_network, data_loaders, device, dir_out, timestamp):
    """
    Train baseline model (supervised learning on source domain only).
    
    Args:
        config: Configuration dictionary
        base_network: Transformer model
        data_loaders: Dictionary with 'source' and 'test' DataLoaders
        device: torch device
        dir_out: Output directory
        timestamp: Timestamp string for file naming
        
    Returns:
        Best model (wrapped in Sequential)
    """
    parameter_list = base_network.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = schedule_dict[optimizer_config["lr_type"]]
    
    len_train_source = len(data_loaders["source"])
    best_acc = 0.0
    best_model_path = None
    training_accuracy = []
    testing_accuracy = []
    classifier_loss_iter = []
    iters = 0
    
    for i in range(config["num_iterations"]):
        base_network.train(True)
        iters += 1
        
        # Evaluate periodically
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            test_result = test(base_network, data_loaders["test"], i, device)
            temp_acc = test_result['accuracy %']

            # Save best model
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model_path = os.path.join(dir_out, f'baseline_best_model_{timestamp}.pth')
                torch.save(base_network.state_dict(), best_model_path)
                logging.info(f"New best baseline model saved with accuracy: {temp_acc:.2f}%")

            testing_accuracy.append((i + 1, temp_acc))
        
        # Training step
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        
        # Get source batch
        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])
        
        inputs_source, labels_source = next(iter_source)
        inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
        
        # Forward pass
        features_source, outputs_source = base_network(inputs_source)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        
        # Backward pass
        classifier_loss.backward()
        optimizer.step()
        classifier_loss_iter.append(classifier_loss.item())
        
        # Compute training accuracy
        preds = outputs_source.argmax(dim=1)
        train_acc = (preds == labels_source).float().mean().item()
        training_accuracy.append((i + 1, train_acc))
        
        # Log periodically
        if (i + 1) % 500 == 0:
            mps_info = (f" | MPS mem: {torch.mps.current_allocated_memory() / 1e6:.0f} MB"
                        if device.type == 'mps' else "")
            logging.info(f'[Iter: {i+1}/{config["num_iterations"]}] Classification loss: {classifier_loss.item():.4f}{mps_info}')
    
    # Plot training curves
    logging.info("Baseline training complete! Plotting training curves...")
    # Safely unpack training and testing accuracy (testing may be empty if no eval occurred)
    if training_accuracy:
        train_iters, train_acc = zip(*training_accuracy)
        train_acc_percentage = [x * 100 for x in train_acc]
    else:
        train_iters, train_acc, train_acc_percentage = [], [], []

    if testing_accuracy:
        test_iters, test_acc = zip(*testing_accuracy)
    else:
        test_iters, test_acc = [], []

    plt.figure()
    if train_iters:
        plt.plot(train_iters, train_acc_percentage, label="Training Accuracy (%)")
    if test_iters:
        plt.plot(test_iters, test_acc, label="Testing Accuracy (%)")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy Curves - Baseline")
    plt.legend()
    training_curve_path = os.path.join(dir_out, f"training_curve_baseline_model_{timestamp}.png")
    plt.savefig(training_curve_path)
    plt.close()
    logging.info(f"Training curve saved to: {training_curve_path}")
    # Return the path to the best saved model (will be loaded in main.py)
    return best_model_path

