# Neural network made with C++ (info below):
This is a neural network for binary sentiment analysis made in C++. The data (Sentiment140 dataset) was cleaned using Python/Jupyter notebook.
All of the paths to files have been changed/omitted because I like keeping my desktop contents secret.


# First Attempt:
<img width="129" alt="Screenshot 2024-02-04 at 9 42 16 PM" src="https://github.com/alexyzha/NeuralNetworkC/assets/122637724/d9e076aa-bdd8-4de3-8e97-648b9ac9eb91">

- 1 hidden layer.
- 50 hidden nodes.
- ReLU for hidden node activation.
- Sigmoid for output normalization.
- Backpropagation is not implemented correctly.
- Atypical node connection.
- 1 file.

# Second Attempt:
<img width="146" alt="Screenshot 2024-02-04 at 9 43 26 PM" src="https://github.com/alexyzha/NeuralNetworkC/assets/122637724/baff5c8a-b155-4ee1-b971-1eaecc61d4e9">

- 2 layers.
- 50 nodes in hidden layer 1.
- 25 nodes in hidden layer 2.
- ReLU for all hidden node activation.
- Sigmoid for output normalization.
- Backpropagation implemented somewhat correctly.
- Atypical node connection.
- 1 file.

# Third Attempt:
<img width="139" alt="Screenshot 2024-02-04 at 9 53 07 PM" src="https://github.com/alexyzha/NeuralNetworkC/assets/122637724/c9b7fc31-ade6-4935-a075-27f9f4c0c2c0">

- 2 layers.
- 25 nodes in hidden layer 1.
- 25 nodes in hidden layer 2.
- ReLU used for all hidden node activation.
- Sigmoid for output normalization.
- Backpropagation implemented correctly.
- ADAM used for backpropagation.
- [Xavier initialization](https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/) used.

