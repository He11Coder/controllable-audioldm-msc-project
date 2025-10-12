import torch

# Loss Functions
def feature_loss(fmap_r, fmap_g):
    """
    Computes the feature matching loss.

    Calculates the L1 distance between discriminator feature maps for real and
    generated audio to encourage structural similarity.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Computes the adversarial loss for the discriminators.

    Uses the LSGAN objective to train discriminators to output 1 for real samples
    and 0 for generated samples.
    """
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
    return loss

def generator_loss(disc_outputs):
    """
    Computes the adversarial loss for the generator.

    Uses the LSGAN objective to train the generator to produce audio that
    the discriminators classify as real (a score of 1).
    """
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        loss += l
    return loss