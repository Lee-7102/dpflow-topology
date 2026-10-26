import torch
import torch.nn as nn


class EuclideanGeometry(nn.Module):
    latent_dim = 2

    def project(self, z):
        return z

    def decode(self, z):
        return z


class DiskGeometry(nn.Module):
    latent_dim = 2

    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self._printed_stats = False

    def project(self, z, eps=1e-8):
        # if not self._printed_stats:
        #     with torch.no_grad():
        #         raw_norm = torch.linalg.vector_norm(z, dim=1)

        #         sat_rate = (raw_norm > 1.0).float().mean()

        #         print(
        #             "disk raw norm:",
        #             "mean", raw_norm.mean().item(),
        #             "p95", torch.quantile(
        #                 raw_norm.flatten(), 0.95
        #             ).item(),
        #             "p99", torch.quantile(
        #                 raw_norm.flatten(), 0.99
        #             ).item(),
        #             "max", raw_norm.max().item(),
        #             "sat", sat_rate.item(),
        #         )

        #     self._printed_stats = True
        norm = torch.linalg.vector_norm(
            z, dim=1, keepdim=True
        )
        # scale = torch.clamp(
        #     1.0 / (norm + eps),
        #     max=1.0,
        # )
        scale = torch.clamp(
            self.radius / (norm + eps),
            max=1.0,
        )
        return z * scale

    def decode(self, z):
        return self.radius * z


class TorusGeometry(nn.Module):
    latent_dim = 4

    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def project(self, z, eps=1e-8):
        zu = z[:, 0:2]
        zv = z[:, 2:4]

        zu = zu / (
            torch.linalg.vector_norm(
                zu, dim=1, keepdim=True
            ) + eps
        )

        zv = zv / (
            torch.linalg.vector_norm(
                zv, dim=1, keepdim=True
            ) + eps
        )

        return torch.cat([zu, zv], dim=1)

    def decode(self, z):
        theta_u = torch.atan2(z[:, 1], z[:, 0])
        theta_v = torch.atan2(z[:, 3], z[:, 2])

        du = self.radius * theta_u / torch.pi
        dv = self.radius * theta_v / torch.pi

        return torch.stack([du, dv], dim=1)

def build_geometry(name, radius):
    if name == "euclidean":
        return EuclideanGeometry()

    if name == "disk":
        return DiskGeometry(radius)

    if name == "torus":
        return TorusGeometry(radius)

    raise ValueError(f"Unknown geometry: {name}")