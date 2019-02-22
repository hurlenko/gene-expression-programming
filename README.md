# Gene expression programming

Simple implementation of Gene expression programming algorithm written in python3. Gene Expression Programming (GEP) is a popular and established evolutionary algorithm for automatic generation of computer programs and mathematical models. .

## Usage

- Install Numpy

```bash
pip install numpy
```

- Tweak constants in `gep.py`, fitness function, number of iterations etc.
- Run

```python
python gep.py
```

## Example

For function `(x * x) + (x * x) - (x + x)` the result is:

```bash
29/200 Current population:
((x + x) * (((((x * x) - x) / x) - x) + x)) = 0.0
```

 > Since GEP doesn't simplify the expressions during evolution, its final result may contain many redundancies, and the tree can be very large, like in this case, which is simply `2(x^2 - x)`. You may like to simplify the final model evolved by GEP with symbolic computation. Take a look at [sympy](https://github.com/sympy/sympy) package.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details