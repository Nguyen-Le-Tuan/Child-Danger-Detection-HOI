"""
clip_encoding.py

CLIP text encoding for interaction labels.
Uses OpenAI's CLIP model (openai/clip-vit-base-patch32) to generate embeddings.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np


# Global model cache
_clip_model = None
_clip_processor = None
_device = None


def get_clip_model():
    """Lazy load CLIP model and processor."""
    global _clip_model, _clip_processor, _device
    
    if _clip_model is None:
        print('[CLIP] Loading model openai/clip-vit-base-patch32...')
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[CLIP] Using device: {_device}')
        
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    
    return _clip_model, _clip_processor, _device


def encode_text(text):
    """
    Encode a single text string using CLIP.
    
    Args:
        text (str): Text to encode
        
    Returns:
        list: CLIP embedding as list of floats
    """
    if not text or not isinstance(text, str):
        return None
    
    model, processor, device = get_clip_model()
    
    try:
        with torch.no_grad():
            inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
            text_features = model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            embedding = text_features.cpu().numpy()[0].tolist()
        return embedding
    except Exception as e:
        print(f'[CLIP] Error encoding text "{text}": {e}')
        return None


def compute_interaction_clip_embeddings(merged_frames):
    """
    Process all frames and compute CLIP embeddings for interaction labels.
    
    Modifies merged_frames in-place:
    - interaction[person_id] changes from string to [string, embedding]
    - e.g., "crawling" -> ["crawling", [0.1, 0.2, ...]]
    
    Args:
        merged_frames (dict): Frame data indexed by frame_id
        
    Returns:
        dict: Same merged_frames with interaction embeddings added
    """
    print('[CLIP] Computing CLIP embeddings for interactions...')
    
    processed_count = 0
    cache = {}  # Cache embeddings for repeated labels
    
    for fid, frame in merged_frames.items():
        for obj in frame.get('objects', []):
            interaction = obj.get('interaction', {})
            
            # interaction is dict {person_id: interaction_type}
            if isinstance(interaction, dict):
                for person_id, interaction_label in interaction.items():
                    # Skip if already encoded
                    if isinstance(interaction_label, list):
                        continue
                    
                    # Use cache to avoid re-encoding same labels
                    if interaction_label not in cache:
                        clip_emb = encode_text(interaction_label)
                        cache[interaction_label] = clip_emb
                    else:
                        clip_emb = cache[interaction_label]
                    
                    # Store as [label, embedding]
                    if clip_emb is not None:
                        obj['interaction'][person_id] = [interaction_label, clip_emb]
                        processed_count += 1
    
    print(f'[CLIP] Processed {processed_count} interaction labels')
    return merged_frames
