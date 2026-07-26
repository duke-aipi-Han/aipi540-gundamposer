# Offline comparison samples

These panels use identical prompts, pose controls, seeds, schedulers, and
generation settings for each pair. The left result disables the LoRA; the right
result enables the selected adapter at strength 0.8.

The three published examples use the public-domain or CC0 source photos listed
in [`../pose-examples/ATTRIBUTION.md`](../pose-examples/ATTRIBUTION.md). The
panels contain only the extracted pose control and generated images.

## Small visual evaluation

One reviewer rated four pose-control examples from 1 to 5. Higher is better for
every metric; artifact quality means fewer visible artifacts. These scores are a
compact project check, not a statistically meaningful benchmark.

| Metric | Base SD 1.5 | Trained LoRA | Difference |
| --- | ---: | ---: | ---: |
| Pose match | 3.25 | 2.50 | -0.75 |
| Mecha appearance | 4.00 | 5.00 | +1.00 |
| Full-body completeness | 4.75 | 4.25 | -0.50 |
| Artifact quality | 3.75 | 3.00 | -0.75 |

The adapter consistently strengthens the custom armor appearance, but the
current 0.8 adapter strength reduces pose fidelity and image cleanliness. The
extreme vertical-kick example is the clearest failure case. This suggests using
a lower adapter strength or more action-oriented training data in future work.

After model warm-up, generation took approximately 10 seconds per image on the
local MPS backend.
