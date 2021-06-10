# TSAI_Session_6
TSAI_Session_6_Assignment

1. Take the last code  (+tweet dataset) and convert that in such a war that:
     1.encoder: an RNN/LSTM layer takes the words in a sentence one by one and finally converts them into a single vector. VERY IMPORTANT TO MAKE THIS SINGLE VECTOR
     2.this single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell
     3.and send this final vector to a Linear Layer and make the final prediction. 
     4.This is how it will look:
        1.embedding
        2.word from a sentence +last hidden vector -> encoder -> single vector
        3.single vector + last hidden vector -> decoder -> single vector
        4.single vector -> FC layer -> Prediction
Your code will be checked for plagiarism, and if we find that you have copied from the internet, then -100%. 
The code needs to look as simple as possible, the focus is on making encoder/decoder classes and how to link objects together
Getting good accuracy is NOT the target, but must achieve at least 45% or more

Once the model is trained, take one sentence, "print the outputs" of the encoder for each step and "print the outputs" for each step of the decoder. ← THIS IS THE ACTUAL ASSIGNMENT
========================================================================================================================================================================
Embedding :  
     Under encoder class embedding was performed as :
     self.embedding = nn.Embedding(vocab_size,embedding_dim)
     
Encoder class: 
class encoders(nn.Module):

  def __init__(self,vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size,embedding_dim)
    self.encoder = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           dropout=dropout,
                           batch_first=True)
    
  def forward(self,text,text_lengths):
    embedded = self.embedding(text)
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
    packed_output, (hidden, cell) = self.encoder(packed_embedded)
    return hidden
    
Decoder class:
class decoders(nn.Module):

  def __init__(self, input_to_decoder_size, decoder_hidden_size, no_times_decoder_cell_has_to_run):
    super().__init__()
    self.decoder_single_rnn_cell = nn.LSTMCell(input_to_decoder_size,decoder_hidden_size)
    self.no_times_decoder_cell_has_to_run = no_times_decoder_cell_has_to_run
    self.decoder_hidden_size = decoder_hidden_size

  def forward(self, encoder_context_vector):
    encoder_context_vector = encoder_context_vector.squeeze()
    hx = torch.zeros(encoder_context_vector.size(0),self.decoder_hidden_size).to(device)
    cx = torch.zeros(encoder_context_vector.size(0),self.decoder_hidden_size).to(device)
    for i in range(self.no_times_decoder_cell_has_to_run):
      hx,cx = self.decoder_single_rnn_cell(encoder_context_vector,(hx,cx))
    return hx
    
 Combined encoder and decoder like this:
 class classifier(nn.Module):
    # Define all the layers used in model
    def __init__(self, encoder, decoder, hidden_dim, output_dim): 
      super().__init__()
      self.encoder = encoder
      self.decoder = decoder
      self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self,src,src_len):
      enc_hidden = self.encoder(src,src_len)
      #print('Encoder: ',enc_hidden)
      dec_hidden = self.decoder(enc_hidden)
      #print('Decoder: ', dec_hidden)
      dense_outputs = self.fc(dec_hidden)
      #print('----',dense_outputs[0])
      #output = F.softmax(dense_outputs,dim=1)
      #print('---',output)
      return dense_outputs
  
Training logs:
	Train Loss: 0.956 | Train Acc: 53.56%
	 Val. Loss: 0.726 |  Val. Acc: 70.19% 

	Train Loss: 0.700 | Train Acc: 69.69%
	 Val. Loss: 0.645 |  Val. Acc: 74.28% 

	Train Loss: 0.400 | Train Acc: 84.04%
	 Val. Loss: 0.731 |  Val. Acc: 80.77% 

	Train Loss: 0.211 | Train Acc: 92.61%
	 Val. Loss: 0.889 |  Val. Acc: 80.05% 

	Train Loss: 0.122 | Train Acc: 97.09%
	 Val. Loss: 1.007 |  Val. Acc: 79.81% 

	Train Loss: 0.084 | Train Acc: 98.23%
	 Val. Loss: 1.025 |  Val. Acc: 80.05% 

	Train Loss: 0.045 | Train Acc: 99.13%
	 Val. Loss: 1.054 |  Val. Acc: 79.81% 

	Train Loss: 0.043 | Train Acc: 98.89%
	 Val. Loss: 1.129 |  Val. Acc: 80.29% 

	Train Loss: 0.031 | Train Acc: 99.20%
	 Val. Loss: 1.224 |  Val. Acc: 77.16% 

	Train Loss: 0.015 | Train Acc: 99.71%
	 Val. Loss: 1.437 |  Val. Acc: 72.36% 
   
For the last requirement, to print the outputs" of the encoder for each step and "print the outputs" for each step of the decoder
-- was not able to achieve.
Faced error in execution:

Encoder:  tensor([[[-2.6472e-04,  1.9274e-02, -6.4385e-02,  2.1966e-01, -1.5182e-01,
          -3.2934e-01, -1.5015e-01,  8.0995e-02, -3.2503e-01, -1.4357e-01,
          -5.1871e-02,  9.7291e-02,  2.6256e-01, -3.0746e-01,  8.1730e-03,
           1.6632e-01,  1.4011e-01,  2.9486e-01, -2.0116e-01, -5.4900e-02,
           1.5134e-01, -7.9561e-03, -2.9733e-01, -2.7189e-01, -3.0584e-01,
           8.2280e-02,  3.6174e-01,  3.0220e-03, -6.0155e-02,  2.5345e-02,
           2.1745e-01,  4.3920e-03,  2.0334e-01, -2.1180e-03, -2.4491e-01,
          -3.3793e-02, -1.9443e-01,  4.7398e-01,  5.4228e-02, -5.6908e-03,
           1.0835e-01,  2.1229e-01,  2.5416e-01,  1.6178e-02, -3.3412e-02,
          -6.8919e-02, -6.0502e-02, -1.1171e-01,  3.4431e-01, -7.3872e-02,
           1.9890e-01, -5.2137e-01, -6.9903e-01, -5.3565e-02,  1.7469e-01,
           2.5398e-01, -2.6382e-01,  6.1046e-02,  2.4883e-02, -5.9994e-02,
           5.5434e-02, -1.7310e-01,  2.4660e-01, -1.9395e-01,  1.6007e-01,
           1.2524e-01, -3.1701e-01,  2.0910e-01,  3.0673e-02, -3.6620e-01,
           9.6695e-02, -5.7842e-02,  1.3749e-01, -1.9586e-01, -3.9776e-01,
          -5.8254e-02, -4.8209e-02,  4.2210e-01,  3.3904e-02,  2.5965e-01,
           1.6108e-02,  1.1796e-02,  3.9525e-01, -3.3428e-02, -2.7006e-01,
          -1.0890e-01, -1.4157e-02,  6.8122e-02, -1.6605e-01, -4.8549e-01,
           2.4384e-01,  4.8924e-03,  8.0320e-02, -1.7115e-01,  9.0012e-02,
           1.5157e-02,  2.8078e-01,  1.2654e-02, -1.4143e-02,  5.8596e-04]]],
       grad_fn=<StackBackward>)
/usr/local/lib/python3.7/dist-packages/torch/nn/modules/rnn.py:63: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.4 and num_layers=1
  "num_layers={}".format(dropout, num_layers))
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-60-25fe155197a0> in <module>()
----> 1 classify_tweet("A valid explanation for why Trump won't let women on the golf course.")

5 frames
/usr/local/lib/python3.7/dist-packages/torch/nn/modules/rnn.py in check_forward_input(self, input)
    870 
    871     def check_forward_input(self, input: Tensor) -> None:
--> 872         if input.size(1) != self.input_size:
    873             raise RuntimeError(
    874                 "input has inconsistent input_size: got {}, expected {}".format(

IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
