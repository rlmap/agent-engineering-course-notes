## Installation

<details>
  <summary>Click to expand for installation instructions if you want to improve the script or run it yourself.</summary>

### Prerequisites
- Python 3.10 or later
- pip (Python package installer)

### Setup Instructions

1. **Create a virtual environment** (from the repository root):
   ```bash
   # Windows
   py -3.10 -m venv --prompt projectgraph venv
   
   # macOS/Linux
   python3.10 -m venv --prompt projectgraph venv
   ```

2. **Activate the virtual environment**:
   ```bash
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Upgrade pip and install base packages**:
   ```bash
   python -m pip install --upgrade pip wheel setuptools
   ```

4. **Install required dependencies**:
   ```bash
   python -m pip install matplotlib seaborn pandas
   ```

### Running the Scripts

Once installed, you can run any of the analysis scripts from the repository root:

```bash
# Run goal extraction
python sidequests/community-projects/project_extraction.py
```

### Dependencies Installed

The installation will include:
- **matplotlib (3.10.3+)**: For creating visualizations and charts
- **seaborn (0.13.2+)**: For enhanced statistical plotting
- **pandas (2.3.0+)**: For data manipulation and analysis
- Additional dependencies: numpy, pillow, fonttools, and others as required
</details>