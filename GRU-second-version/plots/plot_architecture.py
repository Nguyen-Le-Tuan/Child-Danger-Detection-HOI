import matplotlib.pyplot as plt

def box(x, y, w, h, text):
    plt.gca().add_patch(
        plt.Rectangle((x, y), w, h, fill=False)
    )
    plt.text(x + w/2, y + h/2, text, ha="center", va="center")

plt.figure(figsize=(14, 6))
plt.axis("off")

# INPUTS
box(0.5, 2, 2.5, 1, "Numeric Features\n(bbox, distance, velocity)")
box(0.5, 0.5, 2.5, 1, "CLIP Text Embeddings\n(object_type, interaction)")

# ENCODERS
box(4, 2, 2.5, 1, "NumericFeatureEncoder")
box(4, 0.5, 2.5, 1, "CLIPProjector")

# FUSION
box(7.5, 1.25, 2.5, 1.2, "FusionLayer\n(concat + MLP)")

# GRU
box(11, 1.25, 2.5, 1.2, "GRUBackbone\n(temporal modeling)")

# HEADS
box(14.5, 2, 2.8, 1, "Frame Head\n(per-frame logits)")
box(14.5, 0.5, 2.8, 1, "Attention Pooling\n+ ClassifierHead")

# ARROWS
plt.arrow(3, 2.5, 0.8, 0, width=0.02)
plt.arrow(3, 1, 0.8, 0, width=0.02)
plt.arrow(6.8, 2.1, 0.6, -0.4, width=0.02)
plt.arrow(6.8, 1.1, 0.6, 0.4, width=0.02)
plt.arrow(10.3, 1.85, 0.6, 0, width=0.02)
plt.arrow(13.8, 2.2, 0.6, 0, width=0.02)
plt.arrow(13.8, 1.1, 0.6, 0, width=0.02)

plt.title("HOI Danger Detection Model (GRU-based)")
plt.show()
