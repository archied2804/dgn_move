import torch
from .dgn import DiffusionGraphNet
from ..polar_diffusion_process import PolarRadialDiffusionProcess


class PolarDiffusionGraphNet(DiffusionGraphNet):
    """
    DiffusionGraphNet that uses PolarRadialDiffusionProcess for geometry
    denoising. Overrides the reverse sampling step to inject only radial
    noise at each denoising step, matching the forward process geometry.
    """

    @torch.no_grad()
    def sample(self, graph, steps=None, dirichlet_values=None):
        assert isinstance(
            self.diffusion_process, PolarRadialDiffusionProcess
        ), "PolarDiffusionGraphNet requires a PolarRadialDiffusionProcess"

        self.eval()
        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(
                graph.num_nodes, dtype=torch.long, device=self.device)
        if graph.pos.device != self.device:
            graph.to(self.device)

        dp         = self.diffusion_process
        batch_size = graph.batch.max().item() + 1

        theta_init = torch.atan2(graph.pos[:, 1:2], graph.pos[:, 0:1])
        r_noise    = torch.randn(graph.batch.size(0), 1, device=self.device)
        graph.field_r = torch.cat([r_noise * theta_init.cos(),
                                    r_noise * theta_init.sin()], dim=-1)

        step_list = dp.steps[::-1] if steps is None else sorted(steps)[::-1]
        for step in step_list:
            graph.r = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            graph.edge_attr = (graph.field_r[graph.edge_index[1]]
                            - graph.field_r[graph.edge_index[0]])
            model_output = self(graph)
            mean, variance = self.get_posterior_mean_and_variance_from_output(
                model_output, graph, dp)

            if step > 0:
                theta = torch.atan2(graph.field_r[:, 1:2], graph.field_r[:, 0:1])
                z_r   = torch.randn(graph.batch.size(0), 1, device=self.device)
                unit_r = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
                radial_noise = z_r * unit_r
            else:
                radial_noise = torch.zeros_like(mean)

            graph.field_r = mean + torch.sqrt(variance) * radial_noise

        return graph.field_r
