# Inside Airbnb Listings (Napoli)

## University of Prishtina
<img src="https://github.com/user-attachments/assets/9002855f-3f97-4b41-a180-85d1e24ad34a" alt="University Logo" width="110" align="right"/>

**Faculty of Electrical and Computer Engineering (FIEK)**  
**Program:** Computer and Software Engineering - Master  
**Course:** Data Preparation and Visualization  

## Course Professor
**Prof. Dr. Mërgim Hoti**

## Contributors
 Ardi Berdyna, Dafina Keqmezi, Ylljete Kicaj,  Zana Guda

---

## Project Overview
This project focuses on analyzing and visualizing Airbnb listings data for Naples. The goal is to create a clean, structured, and analysis-ready dataset to facilitate insights into listing characteristics, pricing patterns, review trends, and neighborhood-level factors. By preparing and transforming the raw data, this project demonstrates the application of advanced data preprocessing and feature engineering techniques for real-world datasets.

## Project Goals
- Build a complete data preprocessing workflow from raw CSV files to analysis-ready datasets  
- Apply feature engineering techniques to enrich the dataset and improve data quality  
- Perform data aggregation and summary statistics for insightful visualizations  
- Handle missing values, duplicates, and inconsistencies systematically  
- Implement modular and reusable Python scripts for each preprocessing stage  
- Ensure scalability for large datasets while optimizing memory usage  
- Enable downstream analysis for pricing trends, review patterns, and neighborhood insights  

## Dataset Description

The project uses Airbnb listings data for Naples, downloaded from [Inside Airbnb](https://insideairbnb.com/get-the-data/). The dataset consists of **three main files**: `listings.csv`, `reviews.csv`, and `neighbourhoods.csv`.

---

### 1. `listings.csv`  
Contains detailed information about Airbnb listings in Naples, including host details, location, property features, and reviews metrics.  

**Sample attributes and description:**

| Attribute | Description | Example |
|-----------|-------------|---------|
| `id` | Unique identifier for each listing | `63413` |
| `listing_url` | URL of the listing on Airbnb | `https://www.airbnb.com/rooms/63413` |
| `scrape_id` | Unique ID for the data scrape | `20250619031344` |
| `last_scraped` | Date of the last data scrape | `2025-06-19` |
| `source` | Source of the listing data | `city scrape` |
| `name` | Listing title | `Charming Room on Riviera di Chiaia` |
| `description` | Text description of the property | `"Centrally located ... Excellent public transport..."` |
| `neighborhood_overview` | Info about the surrounding neighborhood | `"The house is located in a lively and popular neighborhood..."` |
| `picture_url` | Main listing image | URL |
| `host_id` | Unique ID of the host | `309483` |
| `host_url` | Airbnb profile URL of the host | `https://www.airbnb.com/users/show/309483` |
| `host_name` | Host's name | `Paola` |
| `host_since` | Host registration date | `2010-12-04` |
| `host_location` | Host's reported location | `Campania, Italy` |
| `host_about` | Host description/bio | `"Le Stanze dei Pollano - this is our website"` |
| `host_response_time` | Average response time | `within an hour` |
| `host_response_rate` | Response rate | `100%` |
| `host_acceptance_rate` | Acceptance rate of booking requests | `100%` |
| `host_is_superhost` | Whether host is a superhost | `f` (false) |
| `host_thumbnail_url` | Small host profile picture | URL |
| `host_picture_url` | Medium/large host profile picture | URL |
| `host_neighbourhood` | Host's reported neighborhood | empty |
| `host_listings_count` | Number of listings by host | `1` |
| `host_total_listings_count` | Total listings by host | `2` |
| `host_verifications` | List of verified contact methods | `['email', 'phone']` |
| `host_has_profile_pic` | Host profile picture availability | `t` (true) |
| `host_identity_verified` | Whether host identity is verified | `t` (true) |
| `neighbourhood` | Listing's neighborhood | `Chiaia` |
| `neighbourhood_cleansed` | Cleaned neighborhood name | empty |
| `neighbourhood_group_cleansed` | Broader area (if applicable) | empty |
| `latitude` | Listing latitude | `40.83216` |
| `longitude` | Listing longitude | `14.22642` |
| `property_type` | Type of property | `Entire rental unit` |
| `room_type` | Type of room | `Entire home/apt` |
| `accommodates` | Max guests | `2` |
| `bathrooms` | Number of bathrooms | `1.0` |
| `bathrooms_text` | Bathroom description | `1 bath` |
| `bedrooms` | Number of bedrooms | `1` |
| `beds` | Number of beds | `1` |
| `amenities` | List of amenities | `["Shampoo", "TV", "Wifi", ...]` |
| `price` | Price per night | `$53.00` |
| `minimum_nights` | Minimum nights per booking | `3` |
| `maximum_nights` | Maximum nights per booking | `60` |
| `calendar_updated` | Last update to calendar | `3.3` |
| `has_availability` | Availability status | `t` (true) |
| `availability_30/60/90/365` | Availability in days | `13, 18, 38, 313` |
| `number_of_reviews` | Total reviews received | `132` |
| `number_of_reviews_ltm` | Reviews in last 12 months | `14` |
| `number_of_reviews_l30d` | Reviews in last 30 days | `4` |
| `first_review` | Date of first review | `2013-11-13` |
| `last_review` | Date of last review | `2025-06-02` |
| `review_scores_rating` | Overall rating | `4.64` |
| `review_scores_accuracy` | Accuracy rating | `4.76` |
| `review_scores_cleanliness` | Cleanliness rating | `4.5` |
| `review_scores_checkin` | Check-in rating | `4.91` |
| `review_scores_communication` | Communication rating | `4.83` |
| `review_scores_location` | Location rating | `4.72` |
| `review_scores_value` | Value rating | `4.66` |
| `instant_bookable` | Whether instant booking is allowed | `f` (false) |
| `calculated_host_listings_count` | Total listings by host | `1` |
| `calculated_host_listings_count_entire_homes` | Entire homes listed by host | `1` |
| `calculated_host_listings_count_private_rooms` | Private rooms listed by host | `0` |
| `calculated_host_listings_count_shared_rooms` | Shared rooms listed by host | `0` |
| `reviews_per_month` | Average reviews per month | `0.93` |

---

### 2. `reviews.csv`  
Contains user reviews linked to listing IDs, allowing for time-based analysis of feedback.  

| Attribute | Description | Example |
|-----------|-------------|---------|
| `listing_id` | Listing unique ID | `63413` |
| `id` | Review ID | `8716843` |
| `date` | Review date | `2013-11-13` |
| `reviewer_id` | Reviewer ID | `2811716` |
| `reviewer_name` | Name of reviewer | `Emilie` |
| `comments` | Review text | `"Super séjour. Paola vous dira tout ce qu'il faut faire..."` |

---

### 3. `neighbourhoods.csv`  
Provides a list of neighborhoods in Naples, useful for geographic filtering or mapping.  

| Attribute | Description | Example |
|-----------|-------------|---------|
| `neighbourhood_group` | Optional broader grouping | empty |
| `neighbourhood` | Name of the neighborhood | `Arenella`, `Avvocata` |

---
# Main Pipeline: Airbnb Data Preprocessing and Advanced Analysis

The `main.py` script orchestrates the **entire data preprocessing, integration, cleaning, and advanced feature engineering workflow**. It uses all other modules (`data_collection`, `data_quality`, `integration`, `cleaning`, `advanced_preprocessing`) to produce a fully processed Airbnb dataset ready for analysis or modeling.

---

## Workflow Steps

### 1. Setup Directories
- Determines src, unprocessed, and processed dataset directories.
- Creates the processed dataset directory if it does not exist.

### 2. Data Collection
- Loads datasets (`listings.csv`, `reviews.csv`, `neighbourhoods.csv`) using `load_datasets`.
- Prints loaded datasets and basic row/column counts.
- Handles missing or unexpected files gracefully.

### 3. Define Data Types
- Converts columns to proper types:
  - IDs → strings
  - Prices → numeric
  - Percentages → float
  - Dates → datetime
  - Categories → category
- Samples large datasets for performance (e.g., reviews).

### 4. Data Quality Assessment
- Uses `check_quality` to summarize:
  - Missing values
  - Duplicate rows and IDs
  - Numeric statistics
  - Price outliers (listings)
- Prints quality report summary.

### 5. Integration
- Merges datasets:
  - `merge_neighbourhoods`: Adds `neighbourhood_group` to listings
  - `merge_reviews`: Adds review counts and average review lengths
- Prints row/column counts after each merge.

### 6. Aggregation
- Aggregates listings by a category (e.g., `neighbourhood_cleansed`) using `aggregate_listings`.
- Computes mean price, count, and average rating per group.

### 7. Sampling
- Samples a subset of listings for performance:
  - Stratified sampling by neighbourhood if available
  - Falls back to random sampling if needed

### 8. Cleaning
- Uses `clean_data` to perform:
  - Basic cleaning
  - Removing or correcting invalid values
  - Column-specific adjustments
- Prints updated shape after cleaning.

### 9. Handling Missing Values
- Identifies missing values per column (`identify_missing`).
- Imputes or handles missing values (`handle_missing_values`).
- Compares missing values before and after imputation.

### 10. Advanced Preprocessing
Uses the `AdvancedPreprocessor` class for:
- **Derived Properties:** Creates new features (e.g., `price_log`, ratios, or aggregations)
- **Discretization & Binarization:** Converts continuous features to categorical/binary
- **Property Subset Selection:** Selects high-value or relevant subsets of data
- **Data Transformations:** Applies scaling, normalization, or log transformations
- **Dimension Reduction:** 
  - PCA for numeric feature reduction
  - Univariate feature selection for top features
- Prints shapes and feature summaries at each step.

### 11. Save Processed Data
- Saves final processed listings to CSV.
- Saves PCA-reduced dataset and feature-selected dataset separately.
- Prints final dataset shape, number of columns, and sample newly created features.

---

## Key Notes
- All steps are **logged with timing** to track performance.
- Errors in any step are **handled gracefully** with informative messages.
- Designed to handle **large datasets efficiently** using sampling when necessary.
- Modular design allows for **easy extension** of preprocessing or feature engineering steps.

---

## Example Output
- `integrated_listings.csv` – fully processed dataset
- `listings_pca.csv` – PCA-reduced dataset
- `listings_selected_features.csv` – top selected features
- Summary of new features created during preprocessing is printed to console.
  
## Pipeline Steps and Outputs

### Step 1: Data Collection
- Loads all CSV files from the `unprocessed dataset` folder.
- Datasets loaded:
  - `listings.csv`: 10669 rows, 79 columns  
  - `reviews.csv`: 413675 rows, 6 columns  
  - `neighbourhoods.csv`: 30 rows, 2 columns  
- **Output:** Python dictionaries with pandas DataFrames for each dataset.

### Step 2: Define Data Types
- Converts numerical, categorical, and date columns to correct types.
- Reviews dataset sampled to 100,000 rows for performance.
- **Output:**  
  - Listings: (10669, 79)  
  - Reviews: (100000, 6)  
  - Neighbourhoods: (30, 2)

### Step 3: Data Quality Assessment
- Checks missing values, duplicates, and price outliers in listings.
- **Output:**  
  - Total missing values: 93125  
  - Duplicate IDs: 0  
  - Price outliers: 92  

### Step 4: Integration
- Merges datasets:  
  - `listings` + `neighbourhoods` → adds `neighbourhood_group`  
  - `listings` + `reviews` → adds `avg_review_length` and review-related columns  
- **Output:** Integrated listings DataFrame: (10669, 81)

### Step 5: Aggregation
- Aggregates by neighbourhood: average price, review scores, and counts.
- **Output:** Sample aggregated neighbourhood data (top 10):
  | Neighbourhood | Avg Price | Listings Count | Avg Review Score |
  |---------------|-----------|----------------|----------------|
  | Arenella      | 155.81    | 190            | 4.83           |
  | Avvocata      | 126.49    | 642            | 4.75           |
  | Bagnoli       | 87.41     | 63             | 4.68           |
  | Barra         | 64.82     | 17             | 4.75           |
  | Chiaia        | 184.90    | 729            | 4.75           |

### Step 6: Sampling
- Samples 10% of listings for faster processing.
- **Output:** Sampled listings: 1068 rows  

### Step 7: Cleaning
- Removes duplicate IDs and standardizes values.
- **Output:**  
  - Listings: (1068, 81)  
  - Reviews: adjusted after duplicate removal (5 duplicates removed)

### Step 8: Handle Missing Values
- Imputes numeric columns and fills categorical gaps.
- **Output:**  
  - Total missing values before imputation: 8967  
  - Total missing values after imputation: 4206  

### Step 9: Advanced Preprocessing
- **9.1: Derived Properties**  
  - Created 8 new features: `price_per_person`, `host_experience_years`, `availability_ratio`, `amenity_count`, `bathroom_bedroom_ratio`, `avg_review_score`, `location_desirability`, `is_luxury`  
  - **Output:** (1068, 89)

- **9.2: Discretization & Binarization**  
  - Discretized price, reviews, availability, accommodates  
  - Binarized expensive, many reviews, superhost, instant bookable  
  - **Output:** (1068, 97)  

- **9.3: Property Subset Selection**  
  - Filtered high-value listings (price ≥ $117)  
  - **Output:** (270, 97)  

- **9.4: Data Transformations**  
  - Scaling, log/sqrt transformations, one-hot encoding applied  
  - **Output:** (270, 179)  

- **9.5: Dimension Reduction**  
  - PCA reduced 112 numeric features → 26 components (95% variance)  
  - Univariate selection kept 20 best features  
  - **Output:**  
    - PCA dataset: (270, 93)  
    - Feature-selected dataset: (270, 87)  

- **Preprocessing Summary:**  
  - Original dataset: 1068 rows, 81 columns  
  - Processed dataset: 270 rows, 179 columns  
  - Features added: 98  

### New Features Overview
The 98 new features can be grouped into categories:

| Category | Feature Examples | Count |
|----------|-----------------|-------|
| Binned / Discretized | `price_binned`, `number_of_reviews_x_binned`, `availability_365_binned`, `accommodates_binned` | 8 |
| Scaled / Normalized | `accommodates_scaled`, `beds_scaled`, `price_scaled`, `avg_review_score_scaled` | 32 |
| Derived / Calculated | `price_per_person`, `host_experience_years`, `availability_ratio`, `amenity_count` | 8 |
| Ratio / Interaction | `bathroom_bedroom_ratio`, `availability_ratio_scaled` | 4 |
| Binary / Flags | `is_expensive`, `has_many_reviews`, `is_superhost_binary`, `is_instant_bookable_binary` | 8 |
| One-Hot Encoded | `room_type_*`, `property_type_*`, `neighbourhood_cleansed_*` | 38 |

This categorization provides a quick overview of the types of features added during preprocessing.

### Step 10: Save Processed Data
- Saves final datasets to `processed dataset` folder:  
  - `integrated_listings.csv` (270 rows, 179 columns)  
  - `listings_pca.csv` (PCA-reduced dataset)  
  - `listings_selected_features.csv` (feature-selected dataset)  
- **Output:** Ready-to-use CSV files for analysis or modeling.
