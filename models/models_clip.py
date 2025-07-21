import torch
from torch import nn
import torch.nn.functional as F
import json
import os
from models.clip import clip

def load_clip(args):
    """
    Load the pre-trained CLIP model and disable gradients for inference.
    """
    clip_model, preprocess = clip.load(args.clip_pre_model, 'cpu')
    print("Disabling gradients for CLIP model.")
    for param in clip_model.parameters():
        param.requires_grad_(False)
    return clip_model, preprocess


class ClipSematicModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.clip_model, _ = load_clip(args)
        self.info = self._load_info(args.data_path, args.num_emotion_classes)
        self.args = args

    def _load_info(self, data_root, num_emotion_classes):
        """
        Load dataset metadata and map emotion classes if required.
        """
        assert num_emotion_classes in (8, 2), "Only 8 or 2 emotion classes are supported."
        info = json.load(open(os.path.join(data_root, 'info.json')))
        if num_emotion_classes == 2:
            info['emotion'] = {
                'label2idx': {
                    'amusement': 0, 'awe': 0, 'contentment': 0, 'excitement': 0,
                    'anger': 1, 'disgust': 1, 'fear': 1, 'sadness': 1
                },
                'idx2label': {'0': 'positive', '1': 'negative'}
            }
        return info

    def _generate_prompt(self, keywords, index):
        """
        Construct textual prompts based on provided image attributes.
        """
        prompts = {
            'scene': "The scene type of this photo is {}.",
            'object': "The main subjects are {}.",
            'brightness': "The image has a brightness of {}",
            'colorfulness': "and color saturation of {}.",
            'facial_expression': "Human facial expression is {}.",
            'human_action': "The action being performed is {}.",
            'emotion': "The overall emotion conveyed is {}."
        }

        elements = []
        for key in prompts.keys():
            if key in keywords and keywords[key][index] != '-1':
                value = keywords[key][index]
                if key == 'object':
                    objects = value.split()
                    if len(objects) == 1:
                        value = objects[0]
                    elif len(objects) == 2:
                        value = f"{objects[0]} and {objects[1]}"
                    else:
                        value = f"{', '.join(objects[:-1])} and {objects[-1]}"
                elements.append(prompts[key].format(value))

        emotion_label = self.info['idx2label'][str(keywords['emotion_label_idx'][index].item())]
        if prompts.get('emotion'):
            elements.append(prompts['emotion'].format(emotion_label))

        return " ".join(elements)

    def build_prompts_by_temple(self, keywords):
        """
        Tokenize and encode prompts using CLIP.
        """
        texts = [clip.tokenize(self._generate_prompt(keywords, i)) for i in range(self.args.batch_size)]
        text_features = self.clip_model.encode_text(torch.cat(texts).to(self.args.device, non_blocking=True))
        return text_features / text_features.norm(dim=-1, keepdim=True)


    # def build_prompts_by_temple1(self, keywords):
    #     """
    #     Tokenize and encode prompts using CLIP.
    #     """
    #     text_inputs = self.tokenizer([self._generate_prompt(keywords, i) for i in range(self.args.batch_size)],
    #               return_tensors="pt", padding=True, truncation=True).to("cuda")
    #     text_inputs.pop("token_type_ids", None)
    #     text_features = self.blip_model.text_encoder(**text_inputs).last_hidden_state.mean(dim=1).to(self.args.device,non_blocking=True)
    #     return text_features / text_features.norm(dim=-1, keepdim=True)


    def forward(self, keywords):
        return self.build_prompts_by_temple(keywords)


class ClipBaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.clip_model, _ = load_clip(args)
    def compute_loss(self, image_features, student_image_features, text_features):
        """
        Compute similarity and contrastive loss.
        """
        student_image_features = F.normalize(student_image_features.reshape(len(image_features), -1), p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)

        similarity_loss = 1 - F.cosine_similarity(student_image_features, image_features, dim=1).mean()

        logit_scale = self.clip_model.logit_scale.exp()
        logits_mae = logit_scale * student_image_features @ text_features.t()
        logits_clip = logit_scale * image_features @ text_features.t()

        contrastive_similarity = 1 - F.cosine_similarity(
            F.normalize(logits_mae, p=2, dim=1),
            F.normalize(logits_clip, p=2, dim=1),
            dim=1
        ).mean()

        return similarity_loss, contrastive_similarity

    def contrastive_loss(self, logits):
        """
        Compute contrastive cross-entropy loss.
        """
        return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def siamese_loss(self, similarity):
        """
        Compute symmetric contrastive loss for images and captions.
        """
        return (self.contrastive_loss(similarity) + self.contrastive_loss(similarity.t())) / 2.0

    def forward(self, image, student_image_features, text_features):
        """
        Compute the losses given image, student features, and text features.
        """
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.compute_loss(image_features, student_image_features, text_features)


