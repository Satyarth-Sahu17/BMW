import pytest
import torch
import numpy as np
import os
from src.models.pytorch_models import AudioClassifier
from src.data.ingestion import AudioProcessor
from src.features.augmentation import AudioAugmenter

class TestIntegration:
    @pytest.fixture
    def setup_pipeline(self):
        self.processor = AudioProcessor(sample_rate=32000, duration=5.0)
        self.augmenter = AudioAugmenter(sample_rate=32000)
        self.model = AudioClassifier(num_classes=10, base_model='resnet18')
        
    def test_full_inference_pipeline(self, setup_pipeline):
        # 1. Generate synthetic raw audio
        sr = 32000
        audio = np.random.uniform(-1, 1, sr * 5).astype(np.float32)
        
        # 2. Process audio
        processed = self.processor.process_audio(audio)
        assert processed.shape == (1, 128, 313) # Check mel spec shape
        
        # 3. Augment (optional in inference, but testing flow)
        augmented = self.augmenter.apply_augmentations(audio)
        processed_aug = self.processor.process_audio(augmented)
        assert processed_aug.shape == (1, 128, 313)
        
        # 4. Model inference
        input_tensor = torch.from_numpy(processed).unsqueeze(0) # Add batch dim
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
        assert output.shape == (1, 10)
        assert torch.isclose(probabilities.sum(), torch.tensor(1.0))

    def test_model_save_load(self, tmp_path):
        model = AudioClassifier(num_classes=5)
        save_path = tmp_path / "test_model.pth"
        
        # Save
        torch.save(model.state_dict(), save_path)
        assert os.path.exists(save_path)
        
        # Load
        loaded_model = AudioClassifier(num_classes=5)
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Verify weights match
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)
