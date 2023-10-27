# Advancing Regular Language Reasoning in Linear Recurrent Neural Networks

PyTorch implementation of the paper [Advancing Regular Language Reasoning in Linear Recurrent Neural Networks](https://arxiv.org/abs/2309.07412)

## Installation
* torch >= 2.0 since we used [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
* If torch.compile doesn't work, remove the line of torch.compile in arithmetic.py, pair.py or cycle.py
* That's it!


## Code Run

Modular Arithmetic:
```
python arithmetic.py --seed 0
```

Even Pair:
```
python pair.py --seed 0
```

Sum:
```
python cycle.py --seed 0
```
