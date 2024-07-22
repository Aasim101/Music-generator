import os
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim


def midi_to_notes(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))
    return notes


def encode_notes(notes):
    encoded_notes = [(note[2], note[1] - note[0]) for note in notes]
    return encoded_notes


midi_dir = 'music_dataset'
all_notes = []

for file_name in os.listdir(midi_dir):
    if file_name.endswith('.mid') or file_name.endswith('.midi'):
        file_path = os.path.join(midi_dir, file_name)
        notes = midi_to_notes(file_path)
        all_notes.extend(notes)

encoded_notes = encode_notes(all_notes)

sequence_length = 100
X = []
y = []

for i in range(len(encoded_notes) - sequence_length):
    X.append(encoded_notes[i:i + sequence_length])
    y.append(encoded_notes[i + sequence_length])

X = np.array(X)
y = np.array(y)

max_pitch = min(127, max(note[0] for note in encoded_notes))
max_duration = max(note[1] for note in encoded_notes)

X[:, :, 0] = X[:, :, 0] / max_pitch
X[:, :, 1] = X[:, :, 1] / max_duration
y[:, 0] = y[:, 0] / max_pitch
y[:, 1] = y[:, 1] / max_duration

X = torch.FloatTensor(X)
y = torch.FloatTensor(y)


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


input_size = 2
hidden_size = 256
output_size = 2

model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 50
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def generate_music_rnn(model, seed_sequence, length, max_pitch, max_duration):
    model.eval()
    with torch.no_grad():
        generated_sequence = seed_sequence.clone()
        for _ in range(length):
            input_seq = generated_sequence[-sequence_length:].unsqueeze(0)
            prediction = model(input_seq)
            generated_note = prediction[0].numpy()
            generated_note[0] = int(max(0, min(127, generated_note[0] * max_pitch)))
            generated_note[1] = max(0.1, generated_note[1] * max_duration)
            generated_sequence = torch.cat([generated_sequence, torch.FloatTensor([generated_note])])
    return generated_sequence[sequence_length:].numpy()


seed_sequence = X[0]
generated_sequence = generate_music_rnn(model, seed_sequence, length=500, max_pitch=max_pitch,
                                        max_duration=max_duration)


def sequence_to_midi(sequence, output_file):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Piano
    current_time = 0
    for note in sequence:
        pitch, duration = note
        pitch = int(round(max(0, min(127, pitch))))
        duration = max(0.1, duration)
        start = current_time
        end = start + duration
        midi_note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        instrument.notes.append(midi_note)
        current_time = end
    midi.instruments.append(instrument)
    midi.write(output_file)


sequence_to_midi(generated_sequence, 'generated_music.mid')

print("Music generation complete. Output saved as 'generated_music.mid'")