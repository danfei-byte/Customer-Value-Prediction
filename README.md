## Overview
This project predicts customer-level Average Order Value (AOV) to enable personalized marketing, offer design, and inventory planning. We trained several regression models (Forward Stepwise Linear Regression, k-NN, Decision Tree, Random Forest) on a sampled subset of H&M e-commerce data (transactions, customer attributes, product categories).

**Best model**: Random Forest — highest R² and lowest MAE among tested models.

## Data
- Source: H&M transactional + product + customer tables (The source data can be found [here](https://www.kaggle.com/code/gpreda/h-m-eda-and-prediction/notebook) on Kaggle).
- Sampling: last 1 year of data; random sample of ~50,000 rows for tractability.
- Feature Engineering (summary):
  - **Date integration:** Joined three source tables — transactions, customers, and products — using customer_id and article_id as keys to create a unified customer-level dataset.
  - **Category redefinition:** Re-engineered the product_group_name variable by combining information from product_group_name, product_type_name, and department_name. This ensured that product categories had a clearer granularity, making them easier to interpret and more actionable for analysis (the original groupings were too broad, with dissimilar items collapsed into the same category).
    - e.g., product_type_name with Trousers and Skirt, Shorts, & Tights both fall under same product_group_name "Garment Lower body", making it difficult to distinguish if we keep this original naming convention
  - **Customer-level aggregation:** Aggregated transactions to the customer level to construct modeling features.
  - **Category purchase counts:** Computed purchase counts across 12 consolidated and intuitive product categories (e.g., T-shirts, Trousers, Dresses/Jumpsuits/Sets, Accessories).

The final dataset can be found [here](https://github.com/danfei-byte/Customer-Value-Prediction/blob/1d125bc6ec1f4722f271060de62d61804bb61722/Source%20Data/final_data.csv).
Below is a screenshot of first few rows of the final dataset.
<img src="https://github.com/danfei-byte/Customer-Value-Prediction/blob/e3cff5e6b6f8dd3b9e7ffeea6574e13b3572a906/Sample%20data.png" width="700">

- Target: AOV per customer (total spend / number of orders (per customer)).
- Key features: demographics (age band, club status, fashion news), purchase frequency, category purchase counts across 12 engineered categories (e.g., T-shirts, Trousers, Dresses/Jumpsuits/Sets, Accessories, etc.).

## Exploratory Data Analysis 
Our exploratory analysis focused on understanding customer behavior patterns and product preferences that might influence AOV. We examined customer demographics, engagement metrics, purchase frequency, and spending patterns across different product categories.

Our analysis of spending across product categories reveals distinct purchasing patterns: Categories like T-shirts and Trousers show the highest purchase frequencies. Premium categories like Dresses, Jumpsuits & Sets demonstrate higher per-item spending.
<img src="https://github.com/danfei-byte/Customer-Value-Prediction/blob/a06b24d49b5aaa9dfe987eb95484b24ac7f90c14/EDA.png" width="700">

## Methods (Model Training & Tuning)
We randomly select 80% of the data as our training dataset, and conduct holdout validation to validate model performance on both training and testing dataset.
- Models evaluated:
  - Forward Stepwise Linear Regression (baseline)
  - k-Nearest Neighbors (non-parametric): Tuned 'k' by minimizing MAE; optimal k = 28.
  - Decision Tree (interpretale non-linear): Pruned trees using complexity parameter (Mallow's Cp = 0.00071458) to avoid overfitting.
  - Random Forest (variable importance robust to interactions): Tested number of trees (60 vs 100); 60 trees chosen for computational efficiency with no significant performance loss. 
- Selection criteria: R² (on test), MAE (on test), parsimony/operational fit.

## Results
 <img src="https://github.com/danfei-byte/Customer-Value-Prediction/blob/0561adcc0a4d30e1b296b65d578a5724013b44e8/model%20performance.png" width="700">
 <img src="https://github.com/danfei-byte/Customer-Value-Prediction/blob/dce8b3d97b60e8804c5bb7228a966023df437465/variable%20importance.png" width="700">
Top signals (RF importance): Trousers, T-shirts, Dresses/Jumpsuits/Sets, Accessories, and Age.

## Managerial Implications
Once the Random Forest model was selected as the best performer, I built a scoring pipeline to make the results actionable. The pipeline takes in customer-level data (with the same features used during training) and produces an output table with three key fields:

<image src="https://github.com/danfei-byte/Customer-Value-Prediction/blob/ce8920b786b2cf335265b5629ca837be2b67cac5/Scoring.png" width="400">
<image src="https://github.com/danfei-byte/Customer-Value-Prediction/blob/f81468ee8daf41357c1675584fd973155341a4d9/Distribution.png" width="600">

- aov_pred is the model’s predicted average order value for each customer, i.e., what their next order is likely to look like on average given their history and profile.
- aov_tier is a segmentation bucket (Low / Mid / High), derived from percentile thresholds on predicted AOV (e.g., bottom 40%, middle 40%, top 20%).

### Why this matters for management?
1. **Customer Value Tiering** - *Segment customers into Low/Mid/High groups and align promotions accordingly.*  
   - High-AOV customers can be prioritized for VIP programs, exclusive offers, or early access drops.
   - Low-AOV customers can be nudged upward through bundles, “buy more/save more” offers, or free shipping thresholds. 
2. **Offer Design** - *Use predicted AOV as a natural benchmark for promotions.*
   - Example: if a customer’s predicted AOV is ~$40, management could set a $50 free shipping threshold to encourage larger baskets.
3. **Category-Led Personalization**
   - Because variable importance highlighted categories like Trousers, T-shirts, and Dresses, campaigns can emphasize those product lines for customers with aligned histories.
4. **Inventory Planning**
   - Regions or cohorts with a higher concentration of high-AOV customers in premium categories can be prioritized for new stock, reducing overstock or missed sales opportunities.
5. **Loyalty & CRM Programs**
   - AOV segmentation provides a data-backed way to allocate retention budgets.
   - Rather than treating all customers equally, management can target resources to the groups most likely to generate future revenue.


 
 

 
  
 
  




