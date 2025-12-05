import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet18

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalClassifier, self).__init__()

        # Text encoder (BERT)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(768, 256)  # BERT output is 768-d

        # Image encoder (ResNet18)
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)  # Replace final layer

        # Fusion + classification
        self.fusion = nn.Linear(256 + 256, 128)
        self.classifier = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, image):
        # BERT forward
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = self.text_fc(bert_out.pooler_output)  # shape: (batch, 256)

        # CNN forward
        image_embed = self.cnn(image)  # shape: (batch, 256)

        # Fuse
        combined = torch.cat((text_embed, image_embed), dim=1)
        x = self.relu(self.fusion(combined))
        x = self.dropout(x)
        out = self.classifier(x)
        return out
