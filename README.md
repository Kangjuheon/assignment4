# Assignment 4 – Trustworthy AI

This repository contains my submission for **Assignment #4** of the *Trustworthy AI* course.  
The assignment focuses on using the [α, β-CROWN](https://github.com/Verified-Intelligence/auto_LiRPA) verification tool to evaluate neural network robustness.

---

## 🧪 Objective

- Explore and utilize the **α, β-CROWN** verification tool.
- Run the verifier on an external model and dataset.
- Document the process and observations.

---

## 📂 Repository Structure
```
assignment4/
├── auto_LiRPA/ # α, β-CROWN tool (added as a submodule)
├── configs/ # JSON config files for experiments
├── models/ # Trained models (e.g. MNIST MLP)
├── results/ # Output logs, verification reports
├── run.sh # Script to launch experiment
├── report.pdf # Final 1–2 page write-up
└── README.md # This file


---

## ⚙️ How to Run

1. **Clone this repo with submodules**:
    ```bash
    git clone https://github.com/Kangjuheon/assignment4.git
    cd assignment4
    git submodule update --init --recursive
    ```

2. **Install dependencies**:
    - You can install the required packages manually:
      ```bash
      pip install torch torchvision numpy tqdm matplotlib
      ```

    - Or use the `setup.py` inside the `auto_LiRPA/` submodule:
      ```bash
      cd auto_LiRPA
      pip install -e .
      cd ..
      ```

3. **Run the experiment**:
    ```bash
    bash run.sh
    ```

    Or directly via Python:
    ```bash
    python auto_LiRPA/complete_verifier/patch_runner.py \
      --config configs/mnist_mlp_ab.json \
      --save_path results/mnist_mlp_ab \
      --root models \
      --load_model mnist_mlp.pt
    ```

---

## 📄 Report Contents

See `report.pdf` for a summary of:
- Selected model and dataset
- α, β-CROWN configuration
- Verification results and insights

---

## 📌 Notes

- This project uses [α, β-CROWN](https://github.com/Verified-Intelligence/auto_LiRPA) as a **submodule**.
- Please make sure to initialize submodules as described above.
- The trained model used is a simple MLP on the MNIST dataset.

---

## 🧑‍💻 Author

Juheon Kang (강주헌)  
University of Seoul, 2025  
