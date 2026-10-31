import unittest
import torch
from models.zoo import MODEL_ZOO_REGISTRY, load_zoo_model_class

class TestModelZoo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_x = torch.randn(1, 20, 9, 9)
        cls.dummy_telem = torch.zeros(1, 16)

    def test_all_zoo_models_instantiation_and_forward(self):
        self.assertEqual(len(MODEL_ZOO_REGISTRY), 42, "Model Zoo should have 42 registered models")
        for name in MODEL_ZOO_REGISTRY.keys():
            with self.subTest(model_name=name):
                cls = load_zoo_model_class(name)
                try:
                    model = cls()
                except TypeError:
                    try:
                        model = cls(in_channels=20)
                    except TypeError:
                        model = cls(20)
                
                model.eval()
                with torch.no_grad():
                    try:
                        out = model(self.dummy_x)
                    except TypeError:
                        out = model(self.dummy_x, self.dummy_telem)
                self.assertIsNotNone(out, f"Model {name} returned None output")

if __name__ == "__main__":
    unittest.main()
