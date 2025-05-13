import os, sys
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
import config

class Encoder_Logs:
    def __init__(self, logger):
        self.logger = logger

    def report_weights_and_grads(self, encoder):
    
        for layer_name in ['resonance', 'tension', 'longevity', 'quality', 'combined']:
            
            layer = getattr(encoder, f"{layer_name}_layer")
            if isinstance(layer, nn.Sequential):
                #access the1st layer in the Sequential if it's Linear
                layer_weights = layer[0].weight
            elif isinstance(layer, nn.Embedding):
                layer_weights = layer.weight
            else:
                layer_weights = None

            if layer_weights is not None:
                self.logger.info(f"{layer_name.capitalize()} Layer Weights:")
                self.logger.info(f"  Weight Mean: {layer_weights.mean().item():.6f}")
                self.logger.info(f"  Weight Std: {layer_weights.std().item():.6f}")
                self.logger.info(f"  Weight Min: {layer_weights.min().item():.6f}")
                self.logger.info(f"  Weight Max: {layer_weights.max().item():.6f}")
            else:
                self.logger.info(f"No weights found for {layer_name}_layer.")

            if layer_weights is not None and layer_weights.grad is not None:
                self.logger.info(f"{layer_name.capitalize()} Layer Gradients:")
                self.logger.info(f"  Gradient Mean: {layer_weights.grad.mean().item():.6f}")
                self.logger.info(f"  Gradient Std: {layer_weights.grad.std().item():.6f}")
                self.logger.info(f"  Gradient Min: {layer_weights.grad.min().item():.6f}")
                self.logger.info(f"  Gradient Max: {layer_weights.grad.max().item():.6f}")
            else:
                self.logger.info(f"No gradients for {layer_name}_layer weights.")