import torch.nn as nn

class CRNNWakeWord(nn.Module):
    """
    CRNN + GRU für Wake Word Detection
    Effizientere Variante für Embedded Systeme
    """
    def __init__(self, num_classes=2, conv_channels=64, hidden_size=64, num_layers=2, dropout=0.3, bidirectional=True):
        super(CRNNWakeWord, self).__init__()
        
        # Convolutional Feature Extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, conv_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels // 2),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(conv_channels // 2, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
        )
        
        # Nach Conv: (B, conv_channels, 16, 25)
        self.rnn_input_size = 16 * conv_channels
        
        # GRU statt LSTM
        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)             
        x = x.permute(0, 3, 1, 2)           
        x = x.reshape(batch_size, x.size(1), -1)  
        x, _ = self.gru(x)
        x = x[:, -1, :]                    
        x = self.fc(x)
        return x
