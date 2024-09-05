import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import numpy as np
import time

# Load Inception v3 model pre-trained on ImageNet
class Inception3(nn.Module):
    def __init__(self):
        super(Inception3, self).__init__()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

class FIDCalculator:
    def __init__(self, device):
        self.device = device
        self.model = Inception3().to(device)

    def get_activations(self, images):
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if images.shape[1] == 1:  # Grayscale image
            images = images.repeat(1, 3, 1, 1)  # Convert to RGB
        images = transform(images)
        with torch.no_grad():
            activations = self.model(images).detach()
        return activations

    def calculate_statistics(self, activations):
        mean = torch.mean(activations, dim=0)
        cov = self.torch_cov(activations.T)
        return mean, cov

    def torch_cov(self, m, rowvar=False):
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()
        return fact * m.matmul(mt).squeeze()

    def sqrt_newton_schulz(self, A, numIters):
        with torch.no_grad():
            batchSize = A.shape[0]
            dim = A.shape[1]
            normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
            Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
            I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(A.dtype).to(A.device)
            Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(A.dtype).to(A.device)
            for i in range(numIters):
                T = 0.5 * (3.0 * I - Z.bmm(Y))
                Y = Y.bmm(T)
                Z = T.bmm(Z)
            sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
        return sA

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = self.sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
        return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)

    def calculate_fid(self, real_images, fake_images):
        real_images = real_images.to(self.device)
        fake_images = fake_images.to(self.device)

        real_activations = self.get_activations(real_images)
        fake_activations = self.get_activations(fake_images)

        mu_real, sigma_real = self.calculate_statistics(real_activations)
        mu_fake, sigma_fake = self.calculate_statistics(fake_activations)

        # Check for NaNs or Infs in the statistics
        if torch.isnan(mu_real).any() or torch.isnan(sigma_real).any() or torch.isnan(mu_fake).any() or torch.isnan(sigma_fake).any():
            raise ValueError("NaN values found in statistics")
        if torch.isinf(mu_real).any() or torch.isinf(sigma_real).any() or torch.isinf(mu_fake).any() or torch.isinf(sigma_fake).any():
            raise ValueError("Inf values found in statistics")

        fid = self.calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        torch.cuda.empty_cache()  # Clear unused variables from GPU memory
        return float(torch.clamp(fid, min=0.0))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_calculator = FIDCalculator(device)

    # Test Case 1: Identical Real and Fake Images
    real_images = torch.randn(10, 3, 299, 299, device=device)
    fake_images = real_images.clone()
    start_time = time.time()
    fid_value = fid_calculator.calculate_fid(real_images, fake_images)
    end_time = time.time()
    print(f"FID computation time: {end_time - start_time:.4f} seconds")
    assert fid_value < 0.00001, f"Expected FID to be 0, but got {fid_value}"

    # Test Case 2: Completely Different Real and Fake Images
    real_images = torch.randn(10, 3, 299, 299, device=device)
    fake_images = torch.randn(10, 3, 299, 299, device=device) + 10
    start_time = time.time()
    fid_value = fid_calculator.calculate_fid(real_images, fake_images)
    end_time = time.time()
    print(f"FID computation time: {end_time - start_time:.4f} seconds")
    assert fid_value > 100, f"Expected FID to be high, but got {fid_value}"

    # Test Case 3: Real and Fake Images with Small Perturbations
    real_images = torch.randn(10, 3, 299, 299, device=device)
    fake_images = real_images + 0.1 * torch.randn(10, 3, 299, 299, device=device)
    start_time = time.time()
    fid_value = fid_calculator.calculate_fid(real_images, fake_images)
    end_time = time.time()
    print(f"FID computation time: {end_time - start_time:.4f} seconds")
    assert 0 < fid_value < 10, f"Expected FID to be low, but got {fid_value}"

    # Test Case 4: Real and Fake Images with Different Mean and Variance
    real_images = torch.randn(10, 3, 299, 299, device=device)
    fake_images = 2 * real_images + 1
    start_time = time.time()
    fid_value = fid_calculator.calculate_fid(real_images, fake_images)
    end_time = time.time()
    print(f"FID computation time: {end_time - start_time:.4f} seconds")
    assert fid_value > 10, f"Expected FID to be moderate, but got {fid_value}"

    print("All test cases passed!")