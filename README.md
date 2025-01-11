
# Credit Default Risk Assessment

## Overview

This project is a Python-based solution to analyze and assess credit default risk. It uses a combination of data preprocessing, exploratory data analysis (EDA), and machine learning techniques to predict the likelihood of loan defaults based on customer data.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Dataset Description

The dataset contains key information about loan applicants and their credit history, including:

| Column                       | Description                                      |
|------------------------------|--------------------------------------------------|
| `person_age`                 | Age of the individual.                          |
| `person_income`              | Annual income of the individual.                |
| `person_home_ownership`      | Home ownership status (e.g., RENT, OWN, etc.).  |
| `person_emp_length`          | Length of employment in years.                  |
| `loan_intent`                | Purpose of the loan.                            |
| `loan_grade`                 | Loan grade assigned by the lender.              |
| `loan_amnt`                  | Amount of the loan.                             |
| `loan_int_rate`              | Interest rate of the loan.                      |
| `loan_status`                | Target variable indicating default status (0 or 1). |
| `loan_percent_income`        | Loan amount as a percentage of income.          |
| `cb_person_default_on_file`  | Credit bureau flag for previous defaults.       |
| `cb_person_cred_hist_length` | Length of credit history in years.              |

---

## Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn (optional, for data visualization)

---

## Installation

### Prerequisites

- Python 3.11 or higher
- Pip package manager

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/credit-default-risk-assessment.git
   cd credit-default-risk-assessment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset (`credit_risk_dataset.csv`) is placed in the project directory.

---

## Usage

Run the main script to analyze the data and preprocess it for machine learning:

```bash
python main.py
```

### Outputs

- **Correlation Matrix**: Displays relationships between numeric features.
- **Cleaned Data**: Preprocessed data ready for machine learning.
- **EDA Insights**: Observations from exploratory data analysis.

---

## Project Structure

```
credit-default-risk-assessment/
├── credit_risk_dataset.csv    # Input dataset
├── main.py                    # Main script for analysis and preprocessing
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── results/                   # Outputs from the analysis
└── LICENSE                    # License file
```

---

## Contributing

We welcome contributions! If you'd like to contribute:

1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of your changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a Pull Request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or support, feel free to reach out via GitHub Issues.

---
