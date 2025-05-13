import os, sys
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
import config

class Encoder_File_Management:

    def __init__(self, logger, device):
        self.logger = logger
        self.device = device

    def save_T_encoder_checkpoint(self, model, optimizer, epoch, total_loss, 
                                        epoch_precision_anchor_pos, epoch_precision_anchor_neg, epoch_precision_pos_neg,
                                        epoch_recall_anchor_pos, epoch_recall_anchor_neg, epoch_recall_pos_neg,
                                        epoch_mse_anchor_pos, epoch_mse_anchor_neg, epoch_mse_pos_neg,
                                        quality_computed_accuracy_anchor, quality_computed_accuracy_pos, quality_computed_accuracy_neg,
                                        resonance_mse_anchor, resonance_mse_pos, resonance_mse_neg,
                                        tension_mse_anchor, tension_mse_pos, tension_mse_neg,
                                        longevity_mse_cos_anchor, longevity_mse_cos_pos, longevity_mse_cos_neg,
                                        longevity_mse_sin_anchor, longevity_mse_sin_pos, longevity_mse_sin_neg,
                                        filepath):
        # print("I've entered save ccheckpoint")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'epoch_precision_anchor_pos':epoch_precision_anchor_pos,
            'epoch_precision_anchor_neg':epoch_precision_anchor_neg,
            'epoch_precision_pos_neg':epoch_precision_pos_neg,
            'epoch_recall_anchor_pos':epoch_recall_anchor_pos,
            'epoch_recall_anchor_neg':epoch_recall_anchor_neg,
            'epoch_recall_pos_neg':epoch_recall_pos_neg,
            'epoch_mse_anchor_pos':epoch_mse_anchor_pos,
            'epoch_mse_anchor_neg':epoch_mse_anchor_neg,
            'epoch_mse_pos_neg':epoch_mse_pos_neg,
            'quality_computed_accuracy_anchor': f"{quality_computed_accuracy_anchor:.4f}",
            'quality_computed_accuracy_pos': f"{quality_computed_accuracy_pos:.4f}",
            'quality_computed_accuracy_neg': f"{quality_computed_accuracy_neg:.4f}",
            'resonance_mse_anchor': f"{resonance_mse_anchor:.4f}",
            'resonance_mse_pos': f"{resonance_mse_pos:.4f}",
            'resonance_mse_neg': f"{resonance_mse_neg:.4f}",
            'tension_mse_anchor': f"{tension_mse_anchor:.4f}",
            'tension_mse_pos': f"{tension_mse_pos:.4f}",
            'tension_mse_neg': f"{tension_mse_neg:.4f}",
            'longevity_mse_cos_anchor': f"{longevity_mse_cos_anchor:.4f}",
            'longevity_mse_cos_pos': f"{longevity_mse_cos_pos:.4f}",
            'longevity_mse_cos_neg': f"{longevity_mse_cos_neg:.4f}",
            'longevity_mse_sin_anchor': f"{longevity_mse_sin_anchor:.4f}",
            'longevity_mse_sin_pos': f"{longevity_mse_sin_pos:.4f}",
            'longevity_mse_sin_neg': f"{longevity_mse_sin_neg:.4f}",
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

    def load_T_encoder_checkpoint(self, filepath, model, optimizer):
        self.logger.info(f"Loading Checkpoint from File: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        #load model and opt states
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Extract parameters from checkpoint
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        epoch_auroc = checkpoint.get('epoch_auroc_anchor_pos', None)  # Example handling
        epoch_precision = checkpoint.get('epoch_precision_anchor_pos', None)
        epoch_recall = checkpoint.get('epoch_recall_anchor_pos', None)
        epoch_mse = checkpoint.get('epoch_mse_anchor_pos', None)
        quality_accuracy_1 = checkpoint.get('quality_computed_accuracy_anchor', None)
        quality_accuracy_2 = checkpoint.get('quality_computed_accuracy_pos', None)
        resonance_mse_1 = checkpoint.get('resonance_mse_anchor', None)
        resonance_mse_2 = checkpoint.get('resonance_mse_pos', None)
        tension_mse_1 = checkpoint.get('tension_mse_anchor', None)
        tension_mse_2 = checkpoint.get('tension_mse_pos', None)
        longevity_mse_cos_1 = checkpoint.get('longevity_mse_cos_anchor', None)
        longevity_mse_cos_2 = checkpoint.get('longevity_mse_cos_pos', None)
        longevity_mse_sin_1 = checkpoint.get('longevity_mse_sin_anchor', None)
        longevity_mse_sin_2 = checkpoint.get('longevity_mse_sin_pos', None)
        
        # Print checkpoint details
        self.logger.info(
            f"\nCheckpoint loaded from {filepath}: epoch={epoch}, loss={loss}, "
            f"epoch_auroc={epoch_auroc}, epoch_precision={epoch_precision}, "
            f"epoch_recall={epoch_recall}, epoch_mse={epoch_mse}, "
            f"quality_accuracy_1={quality_accuracy_1}, quality_accuracy_2={quality_accuracy_2}, "
            f"resonance_mse_1={resonance_mse_1}, resonance_mse_2={resonance_mse_2}, "
            f"tension_mse_1={tension_mse_1}, tension_mse_2={tension_mse_2}, "
            f"longevity_mse_cos_1={longevity_mse_cos_1}, longevity_mse_cos_2={longevity_mse_cos_2}, "
            f"longevity_mse_sin_1={longevity_mse_sin_1}, longevity_mse_sin_2={longevity_mse_sin_2}, \n"
        )
        
        return model, optimizer, epoch, loss