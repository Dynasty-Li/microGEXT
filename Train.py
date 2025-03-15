import logging
import math
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from model.RightHand import SpatioTemporalDualNetwork
from model.temp_calibration import TemperatureScaledModel
from utlis import parser
from utlis.pre_data import prepare_dataset
from utlis.skeleton import SkeletonDataset

# Ensure deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def create_data_loaders(config, logger, train_data_path, validation_data_path=None):
    """
    Prepare DataLoaders for training and validation datasets.
    If validation_data_path is not provided, split the train_data into training and validation sets.
    """
    if validation_data_path is None:
        # Prepare dataset and split into training and validation sets
        dataset, labels, states, positives = prepare_dataset(file_pattern=train_data_path, frame_count=config.frame_size)
        logger.info("Splitting data into train and test sets...")
        train_data, val_data, train_labels, val_labels, train_states, val_states, train_positives, val_positives = \
            train_test_split(dataset, labels, states, positives, test_size=0.2, stratify=labels)
    else:
        # Prepare training and validation datasets separately
        logger.info("Segmenting training data from CSV files...")
        train_data, train_labels, train_states, train_positives = prepare_dataset(
            file_pattern=train_data_path, frame_count=config.frame_size)
        logger.info("\nSegmenting validation data from CSV files...")
        val_data, val_labels, val_states, val_positives = prepare_dataset(
            file_pattern=validation_data_path, frame_count=config.frame_size)

    augment = True  # Apply data augmentation if True
    if augment:
        logger.info("\nUsing data noise augmentation")

    # Create SkeletonDataset instances for training and validation
    train_dataset = SkeletonDataset(train_data, train_labels, train_states, train_positives, mode='train', augment=augment)
    val_dataset = SkeletonDataset(val_data, val_labels, val_states, val_positives, mode='test')

    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size, shuffle=True,
                              num_workers=config.workers, pin_memory=False)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size, shuffle=False,
                            num_workers=config.workers, pin_memory=False)

    return train_loader, val_loader


def initialize_model(config, model_class):
    """
    Initialize the model with given configuration and model class.
    """
    # Set model parameters
    num_classes = 8
    num_states = 5
    num_joints = 11
    input_dimension = 7
    save_attention = False

    # Initialize the model instance
    model_instance = model_class(
        num_classes=num_classes,
        num_states=num_states,
        num_joints=num_joints,
        input_dim=input_dimension,
        sequence_length=config.frame_size,
        dropout_rate=config.dp_rate,
        save_attention=save_attention
    )
    # Move model to GPU and wrap in DataParallel
    device = torch.device("cuda")
    model_instance = torch.nn.DataParallel(model_instance).to(device)

    return model_instance


def forward_pass(batch_sample, model, class_criterion, state_criterion):
    """
    Perform a forward pass through the model and compute the loss.
    """
    device = torch.device("cuda")

    # Extract data and move to GPU
    data = batch_sample[0].to(device)
    class_labels = batch_sample[1].type(torch.LongTensor).to(device)
    state_labels = batch_sample[2].type(torch.LongTensor).to(device)
    positive_samples = batch_sample[3].to(device)

    # Forward pass through the model
    class_probabilities, state_probabilities, positive_loss = model(
        data, contrastive=True, positive_pair=positive_samples
    )

    # Compute losses
    loss1 = class_criterion(class_probabilities, class_labels)
    loss2 = state_criterion(state_probabilities, state_labels)
    total_loss = 0.6 * loss1 + 0.2 * loss2 + 0.2 * positive_loss

    return class_probabilities, state_probabilities, total_loss


def get_confusion_matrix(predictions, labels):
    """
    Compute the confusion matrix and accuracy from predictions and true labels.
    """
    device = torch.device('cpu')
    predictions = predictions.to(device).data.numpy()
    labels = labels.to(device).data.numpy()

    # Get predicted class indices
    outputs = np.argmax(predictions, axis=1)

    # Check for NaN values
    if np.any(np.isnan(labels)) or np.any(np.isnan(outputs)):
        print("NaN values found in labels or outputs.")
        print(f"Labels: {labels}")
        print(f"Outputs: {outputs}")

    # Filter out invalid indices
    valid_indices = ~np.isnan(labels) & ~np.isnan(outputs)
    labels = labels[valid_indices]
    outputs = outputs[valid_indices]

    # Compute confusion matrix and accuracy
    matrix = confusion_matrix(labels.flatten(), outputs.flatten(), normalize='true')
    accuracy = accuracy_score(labels.flatten(), outputs.flatten())

    return matrix, accuracy


def plot_confusion_matrix(cm, class_names, figure_path=None):
    """
    Plot and save the confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 16})

    # Change the lable font size
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.title('Confusion Matrix', fontsize=20, pad=20)

    if figure_path:
        plt.savefig(figure_path + '.png', format='png')
        plt.savefig(figure_path + '.svg', format='svg')
        plt.savefig(figure_path + '.pdf', format='pdf')
    else:
        plt.show()
    plt.close()

def create_logger(log_path):
    """
    Create a logger to log information to both console and file.
    """
    # Create a logger object
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)

    # Set up console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Set up file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def train_model(config, model_class, model_name, train_data_path, validation_data_path=None):
    """
    Train the model with the given configuration and data paths.
    """
    if config.training:
        # Create a unique model folder based on current date and model name
        model_name = datetime.now().strftime('%d%m%y') + '_' + model_name
        current_path = os.getcwd()
        model_folder = f"{current_path}/{model_name}"

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # Create a logger
        logger = create_logger(model_folder + "/train.log")

        # Initialize data loaders
        train_loader, val_loader = create_data_loaders(config, logger, train_data_path, validation_data_path)

        # Log input parameters
        logger.info("\nInput Parameters:")
        logger.info(config)

        # Initialize the model
        logger.info("\nInitializing model...")
        model = initialize_model(config, model_class)

        # Set up optimizer and loss functions
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
        class_criterion = torch.nn.CrossEntropyLoss().cuda()
        state_criterion = torch.nn.CrossEntropyLoss().cuda()

        # Get dataset sizes
        train_data_size = len(train_loader.dataset)
        val_data_size = len(val_loader.dataset)
        iterations_per_epoch = math.ceil(train_data_size / config.batch_size)

        # Log training info
        logger.info(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        logger.info(f"Training data size: {train_data_size}")
        logger.info(f"Validation data size: {val_data_size}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Number of workers: {config.workers}")

        # Initialize training variables
        max_accuracy = 0
        no_improve_epoch = 0
        n_iter = 0
        config.epochs = 50
        logger.info("\n*** Starting Training ***")

        # Training loop
        for epoch in range(config.epochs):
            model.train()
            start_time = time.time()
            train_loss = 0

            for i, batch_sample in enumerate(train_loader):
                n_iter += 1
                labels = batch_sample[1]
                state_truth = batch_sample[2]

                # Skip extra batches if any
                if i + 1 > iterations_per_epoch:
                    continue

                # Forward pass and loss computation
                class_probs, state_probs, loss = forward_pass(batch_sample, model, class_criterion, state_criterion)
                train_loss += loss.item()

                # Backpropagation and optimizer step
                model.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate predictions and labels for metrics
                if i == 0:
                    score_list = class_probs
                    label_list = labels
                    state_score = state_probs
                    state_list = state_truth
                else:
                    score_list = torch.cat((score_list, class_probs), 0)
                    label_list = torch.cat((label_list, labels), 0)
                    state_score = torch.cat((state_score, state_probs), 0)
                    state_list = torch.cat((state_list, state_truth), 0)

            # Compute average training loss
            train_loss /= (i + 1)
            # Compute confusion matrices and accuracies
            train_cm, train_acc = get_confusion_matrix(score_list, label_list)
            train_state_cm, train_state_acc = get_confusion_matrix(state_score, state_list)

            # Log training metrics
            logger.info(f"\nEpoch [{epoch + 1}] time: {time.time() - start_time:.4f}, "
                        f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  train_state_acc: {train_state_acc:.4f}")

            # Validation step
            with torch.no_grad():
                val_loss = 0
                model.eval()
                for i, batch_sample in enumerate(val_loader):
                    labels = batch_sample[1]
                    state_truth = batch_sample[2]
                    class_probs, state_probs, loss = forward_pass(batch_sample, model, class_criterion, state_criterion)
                    val_loss += loss.item()

                    # Accumulate predictions and labels for metrics
                    if i == 0:
                        score_list = class_probs
                        label_list = labels
                        state_score = state_probs
                        state_list = state_truth
                    else:
                        score_list = torch.cat((score_list, class_probs), 0)
                        label_list = torch.cat((label_list, labels), 0)
                        state_score = torch.cat((state_score, state_probs), 0)
                        state_list = torch.cat((state_list, state_truth), 0)

                # Compute average validation loss
                val_loss /= (i + 1)
                # Compute confusion matrices and accuracies
                val_cm, val_acc = get_confusion_matrix(score_list, label_list)
                val_state_cm, val_state_acc = get_confusion_matrix(state_score, state_list)

                # Log validation metrics
                logger.info(f"Epoch [{epoch + 1}], val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} val_state_acc: {val_state_acc:.4f}")

                # Plot confusion matrices
                plot_confusion_matrix(val_state_cm, class_names=['State1', 'State2', 'State3', 'State4', 'State5'],
                                      figure_path=f'{model_folder}/epoch_{epoch + 1}_state_confusion_matrix')

                plot_confusion_matrix(val_cm,
                                      class_names=['Fist', 'Cube', 'Vertical', 'Ring', 'Pinky', 'Scissors', 'Swipe', 'Null'],
                                      figure_path=f'{model_folder}/epoch_{epoch + 1}_class_confusion_matrix')

                # Compute total accuracy combining class and state accuracies
                total_accuracy = val_acc * 0.8 + val_state_acc * 0.2

                # Save model if performance improved
                if total_accuracy > max_accuracy:
                    total_accuracy = round(total_accuracy, 3)
                    no_improve_epoch = 0
                    min_loss = val_loss
                    max_accuracy = total_accuracy
                    val_loss = round(val_loss, 3)
                    total_accuracy = round(total_accuracy, 3)
                    best_model_name = f"ep_{epoch + 1}_acc_{total_accuracy}_{datetime.now().strftime('%H%M%S')}"
                    torch.save(model.module.state_dict(), f'{model_folder}/{best_model_name}.pt')
                    logger.info(f"Performance improved, saving new model. Best average accuracy: {total_accuracy}")
                    logger.info(val_cm.diagonal() / val_cm.sum(axis=1))
                else:
                    no_improve_epoch += 1
                    logger.info(f"No_improve_epoch: {no_improve_epoch} Best accuracy: {max_accuracy}")

                # Early stopping if no improvement
                if no_improve_epoch == config.patience:
                    logger.info("*** Stop Training ***")
                    break

    if config.inference:
        print("\n*** Start Inference ***")
        try:
            model_path = f'{model_folder}/{best_model_name}.pt'
        except NameError:
            # If no model was trained, use default model path
            model_folder = f"{os.getcwd()}/191124_microGEXT_model"
            best_model_name = "ep_3_acc_0.987_222104"
            model_path = f'{model_folder}/{best_model_name}.pt'
            logger = create_logger(model_folder + "/train.log")
            # Prepare validation data
            val_data, val_labels, val_states, _ = prepare_dataset(file_pattern=validation_data_path, frame_count=config.frame_size)
            val_dataset = SkeletonDataset(val_data, val_labels, val_states, mode='test')
            val_loader = DataLoader(val_dataset,
                                    batch_size=config.batch_size, shuffle=False,
                                    num_workers=config.workers, pin_memory=False)

        # Initialize model and load weights
        class_criterion = torch.nn.CrossEntropyLoss().cuda()
        state_criterion = torch.nn.CrossEntropyLoss().cuda()
        model = initialize_model(config, model_class)
        model.module.load_state_dict(torch.load(model_path))

        # Wrap model for temperature scaling
        model = TemperatureScaledModel(model).to(torch.device('cuda'))

        with torch.no_grad():
            val_loss = 0
            model.eval()
            for i, batch_sample in enumerate(val_loader):
                labels = batch_sample[1]
                state_truth = batch_sample[2]
                class_probs, state_probs, loss = forward_pass(batch_sample, model, class_criterion, state_criterion)
                val_loss += loss

                # Accumulate predictions and labels for metrics
                if i == 0:
                    score_list = class_probs
                    label_list = labels
                    state_score = state_probs
                    state_list = state_truth
                else:
                    score_list = torch.cat((score_list, class_probs), 0)
                    label_list = torch.cat((label_list, labels), 0)
                    state_score = torch.cat((state_score, state_probs), 0)
                    state_list = torch.cat((state_list, state_truth), 0)

            # Compute average validation loss
            val_loss = val_loss / float(i + 1)
            # Compute confusion matrices and accuracies
            val_cm, val_acc = get_confusion_matrix(score_list, label_list)
            val_state_cm, val_state_acc = get_confusion_matrix(state_score, state_list)
            print(f"Test set loss: {val_loss}")
            print(f"\nClass Accuracy: {val_acc:.4f}\nState Accuracy: {val_state_acc:.4f}")
        return torch.argmax(score_list, dim=1), label_list, val_state_cm

    if config.calibration:
        try:
            model_path = f'{model_folder}/{best_model_name}.pt'
        except NameError:
            # If no model was trained, use default model path
            model_folder = f"{os.getcwd()}/191124_microGEXT_model"
            best_model_name = "ep_3_acc_0.987_222104"
            model_path = f'{model_folder}/{best_model_name}.pt'
            logger = create_logger(model_folder + "/train.log")
            # Prepare data loaders
            train_loader, val_loader = create_data_loaders(config, logger, train_data_path, validation_data_path)
        # Initialize model and load weights
        model = initialize_model(config, model_class)
        model.module.load_state_dict(torch.load(model_path))

        epochs = 5
        scaled_model = TemperatureScaledModel(model)

        # Temperature scaling calibration
        for i in range(epochs):
            scaled_model, before_nll, before_ece, after_nll, after_ece = scaled_model.set_temperature(val_loader)

            logger.info('Before temperature - NLL: %.3f, ECE: %.3f' % (before_nll, before_ece))
            logger.info('After temperature - NLL: %.3f, ECE: %.3f' % (after_nll, after_ece))
            logger.info('Optimal temperature: %.3f' % scaled_model.temperature.item())

        # Save calibrated model
        torch.save(scaled_model.state_dict(), f'{model_folder}/{best_model_name}_scaled.pt')

        print(f"Final Temperature: {scaled_model.temperature.item()}")
        print("\nModel calibrated SUCCESSFULLY!")


if __name__ == "__main__":
    # Parse command-line arguments
    config = parser.parser_args()
    test_subject = '1'
    # Define paths to training and validation data
    training_data_path = f'/training_data/P*[!{test_subject}]/*/*right_hand*.csv'
    validation_data_path = f'/training_data/P*[{test_subject}]/*/*right_hand*.csv'

    # Start the training process
    train_model(config, SpatioTemporalDualNetwork, "microGEXT_model", training_data_path, validation_data_path)
