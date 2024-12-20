import torch
import numpy as np

def add_gaussian_noise_to_model(model_path, output_path, mean=0.0, std=0.01):
    """
    Adds Gaussian noise to the weights of a YOLOv5 model.

    Args:
        model_path (str): Path to the YOLOv5 model file (.pt).
        output_path (str): Path to save the modified model.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        None
    """
    # Load the model
    model = torch.load(model_path)
    model_weights = model['model'].state_dict()

    # Add Gaussian noise to each parameter
    for key in model_weights.keys():
        # Check if the parameter is a tensor and has a dtype of float
        if torch.is_tensor(model_weights[key]) and model_weights[key].dtype == torch.float32:
            noise = torch.normal(mean=mean, std=std, size=model_weights[key].shape)
            model_weights[key] += noise

    # Update the model with the noisy weights
    model['model'].load_state_dict(model_weights)

    # Save the modified model
    torch.save(model, output_path)
    print(f"Model with Gaussian noise saved to {output_path}")

# Input and output paths
model_path = "w_avg.pt"
output_path = "w_avg_noise.pt"

# Add Gaussian noise with mean=0 and std=0.01
add_gaussian_noise_to_model(model_path, output_path)
