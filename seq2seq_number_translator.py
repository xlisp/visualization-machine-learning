import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

import random

class NumeralTranslationDataset:
    def __init__(self):
        # Comprehensive mapping of Arabic numerals to English words
        self.num_to_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', 
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', 
            '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
            '20': 'twenty', '21': 'twenty one', '22': 'twenty two', 
            '23': 'twenty three', '24': 'twenty four', '25': 'twenty five',
            '30': 'thirty', '31': 'thirty one', '32': 'thirty two', 
            '33': 'thirty three', '34': 'thirty four', '35': 'thirty five',
            '40': 'forty', '41': 'forty one', '42': 'forty two', 
            '43': 'forty three', '44': 'forty four', '45': 'forty five',
            '50': 'fifty', '51': 'fifty one', '52': 'fifty two', 
            '53': 'fifty three', '54': 'fifty four', '55': 'fifty five',
            '60': 'sixty', '61': 'sixty one', '62': 'sixty two', 
            '63': 'sixty three', '64': 'sixty four', '65': 'sixty five',
            '70': 'seventy', '71': 'seventy one', '72': 'seventy two', 
            '73': 'seventy three', '74': 'seventy four', '75': 'seventy five',
            '80': 'eighty', '81': 'eighty one', '82': 'eighty two', 
            '83': 'eighty three', '84': 'eighty four', '85': 'eighty five',
            '90': 'ninety', '91': 'ninety one', '92': 'ninety two', 
            '93': 'ninety three', '94': 'ninety four', '95': 'ninety five'
        }

    def generate_training_data(self, num_examples=1000):
        """Generate random training data for number translation."""
        input_sequences = []
        target_sequences = []

        # Generate random numbers from 0 to 99 for a total of `num_examples` examples
        for _ in range(num_examples):
            num = random.randint(0, 99)  # Randomly pick a number from 0 to 99
            num_str = str(num)
            
            # Translate to words
            if num in self.num_to_words:
                word = self.num_to_words[num_str]
            elif num < 20:
                # Handle teens
                units = str(num % 10)
                word = self.num_to_words[units]
            else:
                # Handle 21-99
                tens = str((num // 10) * 10)
                units = str(num % 10)
                tens_word = self.num_to_words[tens]
                units_word = self.num_to_words[units] if units != '0' else ''
                word = f"{tens_word} {units_word}".strip()
            
            input_sequences.append(list(num_str))
            target_sequences.append(list(word))

        return input_sequences, target_sequences

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, hidden dim]
        # encoder_outputs = [batch size, seq len, hidden dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Concatenate hidden state with encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=-1)))
        
        # Compute attention scores
        attention = torch.sum(self.v * energy, dim=-1)
        
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc_out = nn.Linear(hidden_dim, output_size)
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size, 1]
        # hidden = [batch size, hidden dim]
        # encoder_outputs = [batch size, src len, hidden dim]
        
        # Embedding input
        embedded = self.embedding(input)
        
        # Compute attention weights
        a = self.attention(hidden[-1], encoder_outputs)
        
        # Apply attention to encoder outputs
        attended = torch.bmm(a.unsqueeze(1), encoder_outputs).squeeze(1)
        
        # Concatenate embedded input with attended context
        rnn_input = torch.cat((embedded.squeeze(1), attended), dim=1).unsqueeze(1)
        
        # GRU step
        output, hidden = self.gru(rnn_input, hidden)
        
        # Prediction
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, a

class Seq2SeqTranslator(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim):
        super(Seq2SeqTranslator, self).__init__()
        self.encoder = Encoder(input_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(output_size, embedding_dim, hidden_dim)
        
    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        target_vocab_size = self.decoder.fc_out.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        
        # Encoder
        encoder_outputs, hidden = self.encoder(input_seq)
        
        # First decoder input
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)
        
        # Decode
        for t in range(target_len):
            decoder_output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t:t+1, :] = decoder_output.unsqueeze(1)
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            
            if teacher_force:
                decoder_input = target_seq[:, t:t+1]
            else:
                decoder_input = top1.unsqueeze(1)
        
        return outputs

class NumeralTranslator:
    def __init__(self, input_chars, output_chars):
        # Create dataset
        self.dataset = NumeralTranslationDataset()
        
        # Create character to index mappings
        self.input_char_to_idx = {char: i for i, char in enumerate(input_chars)}
        self.input_idx_to_char = {i: char for char, i in self.input_char_to_idx.items()}
        
        self.output_char_to_idx = {char: i for i, char in enumerate(output_chars)}
        self.output_idx_to_char = {i: char for char, i in self.output_char_to_idx.items()}
        
        # Hyperparameters
        self.embedding_dim = 128
        self.hidden_dim = 256
        
        # Initialize model
        self.model = Seq2SeqTranslator(
            input_size=len(input_chars),
            output_size=len(output_chars),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def prepare_sequence(self, seq, char_to_idx):
        """Convert sequence of characters to tensor of indices."""
        return torch.tensor([char_to_idx.get(char, 0) for char in seq], dtype=torch.long)
    
    def pad_sequences(self, sequences, pad_token):
        """Pad sequences to equal length."""
        # Convert sequences to lists if they are tensors
        sequences = [seq.tolist() if torch.is_tensor(seq) else seq for seq in sequences]
        
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [pad_token] * (max_len - len(seq)))
        return torch.tensor(padded, dtype=torch.long)
    
    def train(self, epochs=300, batch_size=32):
        """Train the translation model."""
        # Generate training data
        input_sequences, target_sequences = self.dataset.generate_training_data()
        
        # Prepare input and target sequences
        input_chars = [list(str(seq)) for seq in input_sequences]
        target_chars = [list(seq) for seq in target_sequences]
        
        # Get character sets for input and output
        input_chars_set = sorted(set(''.join([''.join(seq) for seq in input_chars])))
        output_chars_set = sorted(set(''.join([''.join(seq) for seq in target_chars])))
        
        print("Input characters:", input_chars_set)
        print("Output characters:", output_chars_set)
        
        # Training loop
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle data
            combined = list(zip(input_chars, target_chars))
            random.shuffle(combined)
            input_chars, target_chars = zip(*combined)
            
            for i in range(0, len(input_chars), batch_size):
                batch_input = input_chars[i:i+batch_size]
                batch_target = target_chars[i:i+batch_size]
                
                # Prepare input sequences
                input_seqs = self.pad_sequences(
                    [self.prepare_sequence(seq, self.input_char_to_idx) for seq in batch_input], 
                    pad_token=0
                )
                
                # Prepare target sequences
                target_seqs = self.pad_sequences(
                    [self.prepare_sequence(seq, self.output_char_to_idx) for seq in batch_target], 
                    pad_token=0
                )
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_seqs, target_seqs)
                
                # Compute loss
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    target_seqs.view(-1)
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Record average epoch loss
            avg_loss = total_loss / (len(input_chars) // batch_size)
            epoch_losses.append(avg_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        
        # Visualize training loss
        self.plot_training_loss(epoch_losses)
        
        return epoch_losses
    
    def translate(self, input_number):
        """Translate a single number to words."""
        # Prepare input sequence
        input_seq = self.prepare_sequence(list(str(input_number)), self.input_char_to_idx)
        input_seq = input_seq.unsqueeze(0)  # Add batch dimension
        
        # Create dummy target sequence of zeros
        max_output_length = 10  # Maximum expected word length
        dummy_target = torch.zeros(1, max_output_length, dtype=torch.long)
        
        # Disable gradient computation
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_seq, dummy_target)
            
            # Get the most likely output characters
            predicted_indices = outputs.argmax(dim=-1)
            
            # Convert indices back to characters
            predicted_chars = []
            for i in range(predicted_indices.size(1)):
                char_idx = predicted_indices[0, i].item()
                char = self.output_idx_to_char[char_idx]
                if char != '<pxad>':  # Skip padding
                    predicted_chars.append(char)
            
            # Join characters to form a word
            return ''.join(predicted_chars).strip()
    
    def save_model(self, filepath='numeral_translator.pth'):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_char_to_idx': self.input_char_to_idx,
            'output_char_to_idx': self.output_char_to_idx
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='numeral_translator.pth'):
        """Load model state."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_char_to_idx = checkpoint['input_char_to_idx']
        self.output_char_to_idx = checkpoint['output_char_to_idx']
        print(f"Model loaded from {filepath}")
    
    def plot_training_loss(self, losses):
        """Visualize training loss."""
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_loss.png')
        plt.close()

def main():
    # Define input and output character sets
    input_chars = list('0123456789')
    output_chars = list(' abcdefghijklmnopqrstuvwxyz') + ['<pad>']

    # Initialize translator
    translator = NumeralTranslator(input_chars, output_chars)

    # Train the model
    print("Training model...")
    losses = translator.train(epochs=300, batch_size=32)

    # Save the trained model
    translator.save_model()

    # Test the model with some examples
    test_numbers = ['0', '5', '13', '25', '42', '67', '89', '99']
    print("\nTesting translations:")
    for number in test_numbers:
        translation = translator.translate(number)
        print(f"{number} -> {translation}")

    # Interactive mode
    print("\nEnter a number (0-99) to translate or 'q' to quit:")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break
        try:
            number = int(user_input)
            if 0 <= number <= 99:
                translation = translator.translate(user_input)
                print(f"Translation: {translation}")
            else:
                print("Please enter a number between 0 and 99")
        except ValueError:
            print("Invalid input. Please enter a valid number or 'q' to quit")

if __name__ == "__main__":
    main()

