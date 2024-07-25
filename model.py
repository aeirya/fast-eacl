import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False, device='cpu'):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size  # Size of the hidden state
        self.input_size = input_size    # Size of the input features
        
        self.cuda_flag = cuda_flag      # Flag to indicate if CUDA (GPU) is used
        self.device = device            # Device to use ('cpu' or 'cuda')
        
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)  # Linear layer for hidden state transformation
        self.U_all = nn.Linear(input_size, hidden_size * 4)   # Linear layer for input transformation
        self.W_d = nn.Linear(hidden_size, hidden_size)        # Linear layer for discounting short-term memory
        
        self.bidirectional = bidirectional  # Flag to indicate if the LSTM is bidirectional (not used in this implementation)

    def forward(self, input_sequence, timestamps, reverse=False):
        batch_size, seq, embedding = input_sequence.size()  # Get the batch size, sequence length, and input embedding size

        # Initialize hidden and cell states
        hidden_state_t = torch.zeros(batch_size, self.hidden_size, requires_grad=False)
        cell_state_t = torch.zeros(batch_size, self.hidden_size, requires_grad=False)
        
        # # Move states to the appropriate device
        # if self.cuda_flag:
        #     hidden_state_t = hidden_state_t.cuda()
        #     cell_state_t = cell_state_t.cuda()
        # else:
        #     hidden_state_t = hidden_state_t.to(self.device)
        #     cell_state_t = cell_state_t.to(self.device)

        hidden_state_t = hidden_state_t.to(self.device)
        cell_state_t = cell_state_t.to(self.device)

        outputs = []  # List to store the output at each time step (output sequence)
        all_hidden_states = []  # List to store hidden states
        all_cell_states = []  # List to store cell states (hidden_state_c)

        for s in range(seq):  # Loop over each time step in the sequence
            c_s1 = torch.tanh(self.W_d(cell_state_t))  # Compute short-term memory component
            c_s2 = c_s1 * timestamps[:, s: s + 1].expand_as(c_s1)  # Discount short-term memory using timestamps
            c_l = cell_state_t - c_s1  # Compute long-term memory component
            c_adj = c_l + c_s2  # Adjusted cell state combining long-term and discounted short-term memory
            
            # Compute the output of the gates and candidate cell state
            gate_inputs = self.W_all(hidden_state_t) + self.U_all(input_sequence[:, s])
            forget_gate, input_gate, out_gate, c_candidate = torch.chunk(gate_inputs, 4, 1)
            forget_gate = torch.sigmoid(forget_gate)  # Forget gate
            input_gate = torch.sigmoid(input_gate)  # Input gate
            out_gate = torch.sigmoid(out_gate)  # Output gate
            c_candidate = torch.sigmoid(c_candidate)  # Candidate cell state (c_tmp)

            # Update the cell state and hidden state
            cell_state_t = forget_gate * c_adj + input_gate * c_candidate
            hidden_state_t = out_gate * torch.tanh(cell_state_t)

            outputs.append(out_gate)  # Store the output of the current time step
            all_cell_states.append(cell_state_t)  # Store the cell state of the current time step
            all_hidden_states.append(hidden_state_t)  # Store the hidden state of the current time step
        
        if reverse:
            outputs.reverse()
            all_cell_states.reverse()
            all_hidden_states.reverse()
        
        # Stack the outputs and hidden/cell states to form tensors
        outputs = torch.stack(outputs, 1)
        all_cell_states = torch.stack(all_cell_states, 1)
        all_hidden_states = torch.stack(all_hidden_states, 1)

        return outputs, (hidden_state_t, cell_state_t)  # Return the outputs and the final hidden/cell states


    def to(self, device):
        self.device = device
        self.cuda_flag = device == 'cuda'
        
        super(TimeLSTM, self).to(device)
        return self
    

class Attention(torch.nn.Module):
    def __init__(self, in_shape, use_attention=True, maxlen=None):
        super(Attention, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = torch.nn.Linear(in_shape, in_shape)
            self.W2 = torch.nn.Linear(in_shape, in_shape)
            self.V = torch.nn.Linear(in_shape, 1)
        if maxlen != None:
            self.arange = torch.arange(maxlen)
    def forward(self, full, last, lens=None, dim=1):
        if self.use_attention:
            score = self.V(torch.tanh(self.W1(last) + self.W2(full)))
            attention_weights = F.softmax(score, dim=dim)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector
        else:
            return torch.mean(full, dim=dim)


class FAST(nn.Module):
    def __init__(self, num_stocks):
        super(FAST, self).__init__()
        self.num_stocks = num_stocks  # Number of stocks

        # Initialize LSTM layers for text processing for each stock
        self.text_lstm_layers = [nn.LSTM(768, 64) for _ in range(num_stocks)]
        for i, text_lstm in enumerate(self.text_lstm_layers):
            self.add_module(f'text_lstm_{i}', text_lstm)

        # Initialize TimeLSTM layers for each stock
        self.time_lstm_layers = [TimeLSTM(768, 64) for _ in range(num_stocks)]
        for i, time_lstm in enumerate(self.time_lstm_layers):
            self.add_module(f'time_lstm_{i}', time_lstm)

        # Initialize LSTM layers for day processing for each stock
        self.day_lstm_layers = [nn.LSTM(64, 64) for _ in range(num_stocks)]
        for i, day_lstm in enumerate(self.day_lstm_layers):
            self.add_module(f'day_lstm_{i}', day_lstm)

        # Initialize Attention layers for text processing for each stock
        self.text_attention_layers = [Attention(64, 10) for _ in range(num_stocks)]
        for i, text_attention in enumerate(self.text_attention_layers):
            self.add_module(f'text_attention_{i}', text_attention)

        # Initialize Attention layers for day processing for each stock
        self.day_attention_layers = [Attention(64, 5) for _ in range(num_stocks)]
        for i, day_attention in enumerate(self.day_attention_layers):
            self.add_module(f'day_attention_{i}', day_attention)

        # Linear layer for stock prediction
        self.stock_prediction_layer = nn.Linear(64, 1)


    def forward(self, text_inputs, time_inputs):
        num_stocks = self.num_stocks
        stock_outputs = []  # List to store outputs for each stock
        lstm_output_size = 64  # Output size of the LSTM layers

        # Process each stock individually
        for stock_idx in range(text_inputs.size(0)):
            day_outputs = []  # List to store daily outputs for the current stock
            lookback_window_length = text_inputs.size(1)
            num_text_inputs = text_inputs.size(2)
            embedding_dim = text_inputs.size(3)

            # Process each day in the lookback window
            for day_idx in range(lookback_window_length):
                text_input_reshaped = text_inputs[stock_idx, day_idx, :, :].reshape(1, num_text_inputs, embedding_dim)
                time_input_reshaped = time_inputs[stock_idx, day_idx, :].reshape(1, num_text_inputs)

                # Pass through TimeLSTM
                lstm_output, (hidden_state, _) = self.time_lstm_layers[stock_idx](text_input_reshaped, time_input_reshaped)
                # Apply attention on the LSTM output
                attention_output = self.text_attention_layers[stock_idx](lstm_output, hidden_state, num_text_inputs)
                
                day_outputs.append(attention_output)

            # Combine daily outputs and process through day LSTM
            day_outputs_tensor = torch.cat(day_outputs).reshape(1, lookback_window_length, lstm_output_size)
            day_lstm_output, (hidden_state_day, _) = self.day_lstm_layers[stock_idx](day_outputs_tensor)
            # Apply attention on the day LSTM output
            final_attention_output = self.day_attention_layers[stock_idx](day_lstm_output, hidden_state_day, lookback_window_length)
            stock_outputs.append(final_attention_output.reshape(1, lstm_output_size))


        # Combine outputs for all stocks and pass through final linear layer
        final_output_tensor = torch.cat(stock_outputs)
        final_output = F.leaky_relu(self.stock_prediction_layer(final_output_tensor))

        print("final_output.shape: ", final_output.shape)
        print("final_output: ", final_output)
        # quit(0)

        return final_output
    
    def to(self, device):
        
        for item in self.time_lstm_layers:
            item.to(device)
        
        super(FAST, self).to(device)
        return self
