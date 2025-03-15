import torch
from torch import nn, optim
from torch.nn import functional as F


class TemperatureScaledModel(nn.Module):
    """
    A wrapper model that applies temperature scaling to a classification network.
    Note: The base_model should output logits, not probabilities.
    """

    def __init__(self, base_model):
        super(TemperatureScaledModel, self).__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, inputs):
        logits, states = self.base_model(inputs)
        scaled_logits = self._apply_temperature_scaling(logits)
        return scaled_logits, states

    def _apply_temperature_scaling(self, logits):
        """
        Apply temperature scaling to logits.
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def calibrate_temperature(self, validation_loader):
        """
        Calibrate the temperature parameter using a validation dataset.
        The temperature is optimized to minimize the negative log-likelihood (NLL).
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ExpectedCalibrationError().cuda()

        # Collect logits and labels from the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, class_labels, state_labels in validation_loader:
                inputs = inputs.cuda()
                logits, states = self.base_model(inputs)
                logits_list.append(logits)
                labels_list.append(class_labels)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_nll = nll_criterion(logits, labels).item()
        before_ece = ece_criterion(logits, labels).item()

        # Optimize the temperature parameter
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=50)

        def eval_func():
            optimizer.zero_grad()
            loss = nll_criterion(self._apply_temperature_scaling(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_func)

        # Calculate NLL and ECE after temperature scaling
        after_nll = nll_criterion(self._apply_temperature_scaling(logits), labels).item()
        after_ece = ece_criterion(self._apply_temperature_scaling(logits), labels).item()

        print('Optimal temperature: {:.3f}'.format(self.temperature.item()))
        print('Before temperature scaling - NLL: {:.3f}, ECE: {:.3f}'.format(before_nll, before_ece))
        print('After temperature scaling - NLL: {:.3f}, ECE: {:.3f}'.format(after_nll, after_ece))

        return self, before_nll, before_ece, after_nll, after_ece


class ExpectedCalibrationError(nn.Module):


    def __init__(self, n_bins=15):

        super(ExpectedCalibrationError, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lower_bounds = bin_boundaries[:-1]
        self.bin_upper_bounds = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmax_probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmax_probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for lower, upper in zip(self.bin_lower_bounds, self.bin_upper_bounds):
            # Calculate |confidence - accuracy| in each bin
            in_bin = confidences.gt(lower.item()) * confidences.le(upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
