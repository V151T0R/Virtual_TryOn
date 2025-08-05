import torch

class LoadModel:
    def __init__(self, model_instance, model_path, device=None):
        """
        Args:
            model_instance (nn.Module):  Instantiated model.
            model_path (str): Path to the saved model weights.
            device (torch.device, optional): Device to load model on.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_instance.to(self.device)
        self.loaded = False

        try:
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode
            self.loaded = True
        except Exception as e:
            print(" Failed to load model: {e}")
            self.loaded = False

    def get_model(self):
        """Returns the model (in eval mode)."""
        return self.model

    def is_loaded(self):
        """Returns True if model loaded successfully, else False."""
        return self.loaded
