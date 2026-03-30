import torch
from typing import Tuple
from .diffusion_process import DiffusionProcess


class PolarRadialDiffusionProcess(DiffusionProcess):
    """
    Forward diffusion that adds noise only to the radial component r.
    At full noise (t=T), r ~ N(0,1) and theta is preserved, giving
    a cloud that is Gaussian along each radial ray from the origin.

    Assumes input positions are 2D Cartesian, centred at (0,0).
    """


    @staticmethod
    def xy_to_polar(xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[N,2] -> r:[N,1], theta:[N,1]"""
        r     = torch.norm(xy, dim=-1, keepdim=True)          # [N,1]
        theta = torch.atan2(xy[:, 1:2], xy[:, 0:1])           # [N,1]
        return r, theta

    @staticmethod
    def polar_to_xy(r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """r:[N,1], theta:[N,1] -> [N,2]"""
        return torch.cat([r * torch.cos(theta),
                          r * torch.sin(theta)], dim=-1)


    def __call__(
        self,
        field_start:    torch.Tensor,   # [N, 2]  clean positions
        r:              torch.Tensor,   # [B]     diffusion step index per graph
        batch:          torch.Tensor,   # [N]
        dirichlet_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r""" Forwards the polar radial diffusion process from 'field_start' to diffusion-step `r`.

        Args:
            field_start (torch.Tensor): The initial field defined on the nodes of a graph. Dimensions
                [num_nodes, 2] (2D Cartesian positions).
            r (torch.Tensor): The number of diffusion steps to perform. Dimensions: [batch_size].
            batch (torch.Tensor): The batch indices of the nodes of the graph. Dimensions: [num_nodes]. 
                Defaults to 'None'. If 'None', then it is assumed that all nodes belong to the same graph.
            dirichlet_mask (torch.Tensor, optional): A mask that indicates which nodes have a Dirichlet boudnary
                condition. Dimensions: [num_nodes, 2]. Wherever the mask is 1, the field is not diffused.
                If 'None', then it is assumed that there are no Dirichlet boundary conditions. Defaults to 'None'.

        Returns:
            torch.Tensor: The field after 'r' diffusion steps, defined on the nodes of a graph. Dimensions: [num_nodes, 2].
            torch.Tensor: The (normalised Gaussian) noise employed to diffuse 'field_start'. Dimensions: [num_nodes, 2].    
        
        """
        # Convert to polar coordinates
        r0, theta = self.xy_to_polar(field_start)   # [N,1], [N,1]

        # Sample noise in the radial direction only (i.e. 1D noise per node)
        eps_r = torch.randn_like(r0)                # 1-D radial noise [N,1]

        # DDPM coefficients indexed per node from the batch step
        sqrt_ac   = self.get_index_from_list(self.sqrt_alphas_cumprod,           batch, r)  # [N,1]
        sqrt_1mac = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, batch, r)  # [N,1]

        # Diffuse the radial component only using the DDPM formula, keeping theta fixed
        r_noisy = sqrt_ac * r0 + sqrt_1mac * eps_r  # [N,1]

        # Convert back to Cartesian coordinates
        field_noisy = self.polar_to_xy(r_noisy, theta)  # [N,2]

        # 2-D noise vector — only the radial component is non-zero
        # noise_2d = eps_r * unit_radial_vector
        unit_r  = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)  # [N,2]
        noise_2d = eps_r * unit_r                                           # [N,2]

        if dirichlet_mask is not None:
            field_noisy = torch.where(dirichlet_mask, field_start, field_noisy)
            noise_2d    = noise_2d * (~dirichlet_mask).float()

        return field_noisy, noise_2d