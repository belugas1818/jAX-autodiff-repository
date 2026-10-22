"""Optional matplotlib dump of real vs generated samples."""

from __future__ import annotations


def save_plot(path, spec, real, samples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = spec.denormalize(real)
    samples = spec.denormalize(samples)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if spec.shape == (2,):
        axes[0].scatter(real[:, 0], real[:, 1], s=2)
        axes[1].scatter(samples[:, 0], samples[:, 1], s=2, color="tab:orange")
        for ax in axes:
            ax.set_aspect("equal")
    else:
        flat_r = real.reshape(real.shape[0], -1)
        flat_s = samples.reshape(samples.shape[0], -1)
        for i in range(min(8, flat_r.shape[0], flat_s.shape[0])):
            axes[0].plot(flat_r[i], lw=1)
            axes[1].plot(flat_s[i], lw=1)
    axes[0].set_title("real data")
    axes[1].set_title("diffusion samples")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
