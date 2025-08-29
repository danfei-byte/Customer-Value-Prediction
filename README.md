## Overview
This project predicts customer-level Average Order Value (AOV) to enable personalized marketing, offer design, and inventory planning. We trained several regression models (Forward Stepwise Linear Regression, k-NN, Decision Tree, Random Forest) on a sampled subset of H&M e-commerce data (transactions, customer attributes, product categories).

**Best model**: Random Forest — highest R² and lowest MAE among tested models.

## Data
- Source: H&M transactional + product + customer tables (The source data can be found [here](https://www.kaggle.com/code/gpreda/h-m-eda-and-prediction/notebook) on Kaggle).
- Sampling: last 1 year of data; random sample of ~50,000 rows for tractability.
- Feature Engineering (summary):
  - **Date integration:** Joined three source tables — transactions, customers, and products — using customer_id and article_id as keys to create a unified customer-level dataset.
  - **Category redefinition:** Re-engineered the product_group_name variable by combining information from product_group_name, product_type_name, and department_name. This ensured that product categories had a clearer granularity, making them easier to interpret and more actionable for analysis (the original groupings were too broad, with dissimilar items collapsed into the same category).
    - e.g., product_type_name with Trousers and Skirt, Shorts, & Tights both fall under same product_group_name "Garment Lower body", making it difficult to distinguish if we keep this original naming convention)
  - **Customer-level aggregation:** Aggregated transactions to the customer level to construct modeling features.
  - **Category purchase counts:** Computed purchase counts across 12 consolidated and intuitive product categories (e.g., T-shirts, Trousers, Dresses/Jumpsuits/Sets, Accessories).

The final dataset can be found [here](https://github.com/danfei-byte/Customer-Value-Prediction/blob/1d125bc6ec1f4722f271060de62d61804bb61722/Source%20Data/final_data.csv).

- Target: AOV per customer (total spend / number of orders (per customer)).
- Key features: demographics (age band, club status, fashion news), purchase frequency, category purchase counts across 12 engineered categories (e.g., T-shirts, Trousers, Dresses/Jumpsuits/Sets, Accessories, etc.).

## Exploratory Data Analysis 

## Methods
We randomly select 80% of the data as our training dataset, and conduct holdout validation to validate model performance on both training and testing dataset.
- Modle evaluated:
  - Forward Stepwise Linear Regression (baseline)
  - k-Nearest Neighbors (non-parametric)
  - Decision Tree (interpretale non-linear)
  - Random Forest (variable importance robust to interactions)
- Selection criteria: R² (on test), MAE (on test), parsimony/operational fit.

## Results
 <img src="https://github.com/danfei-byte/Customer-Value-Prediction/blob/0561adcc0a4d30e1b296b65d578a5724013b44e8/model%20performance.png" width="700">
 <img src="https://github.com/danfei-byte/Customer-Value-Prediction/blob/dce8b3d97b60e8804c5bb7228a966023df437465/variable%20importance.png" width="700">
Top signals (RF importance): Trousers, T-shirts, Dresses/Jumpsuits/Sets, Accessories, and Age.
 
 

 
  
 
  




