# quantum-exploration

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/youjunliuhb/quantum-exploration.git
cd quantum-exploration
```

First, create a virtual environment named TheEnv for a dedicated exploration environment:

```bash
python -m venv TheEnv
```

Activate the virtual environment:
- On Windows: `TheEnv\Scripts\activate`
- On macOS/Linux: `source TheEnv/bin/activate`

Install the required dependencies using pip:

```bash
pip install numpy scipy matplotlib pandas seaborn pillow "qiskit[visualization]"
```