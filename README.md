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

## Technologies Used
- **Python 3.11** – Primary programming language  
- **pandas, numpy** – Data manipulation and numerical operations  
- **scikit-learn** – Feature engineering, preprocessing, PCA, and outlier detection (Isolation Forest, LOF)  
- **scipy** – Statistical functions (Z-score)  
- **OS / pathlib** – File and path handling  
- **datetime, re** – Preprocessing dates and text  

---

## Installation and Setup

1. **Clone the repository**

```bash
git clone https://github.com/ylljetakicaj/PVDH_G12.git
cd PVDH_G12
```
2. **Install required packages**
```bash
pip install -r src/requirements.txt
``` 
3. **Run the main pipeline**
```bash
python src/main.py
``` 

1. **Clone the repository**

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

## Main Pipeline: Data Preprocessing and Advanced Analysis

The `main.py` script orchestrates the **entire data preprocessing, integration, cleaning, advanced feature engineering, exploratory data analysis, and outlier detection workflow**. It uses all other modules (data_collection, data_quality, integration, cleaning, advanced_preprocessing, eda_analyzer, outlier_detection) to produce a fully processed Airbnb dataset ready for analysis or modeling.

### Workflow Steps

#### 1. Setup Directories
- Determines src, unprocessed, and processed dataset directories
- Creates the processed dataset directory if it does not exist

#### 2. Data Collection
- Loads datasets (listings.csv, reviews.csv, neighbourhoods.csv) using `load_datasets`
- Prints loaded datasets and basic row/column counts
- Handles missing or unexpected files gracefully

**Initial Dataset Sizes:**
- listings.csv: 10,669 rows, 79 columns
- reviews.csv: 413,675 rows, 6 columns
- neighbourhoods.csv: 30 rows, 2 columns

#### 3. Define Data Types
- Converts columns to proper types:
  - IDs → strings
  - Prices → numeric
  - Percentages → float
  - Dates → datetime
  - Categories → category
- Samples large datasets for performance (e.g., reviews sampled to 100,000 rows)

**After Type Conversion:**
- Listings: (10,669, 79)
- Reviews: (100,000, 6)
- Neighbourhoods: (30, 2)

#### 4. Data Quality Assessment
- Uses `check_quality` to summarize:
  - Missing values (Total: 93,125)
  - Duplicate rows and IDs (Duplicate IDs: 0)
  - Numeric statistics
  - Price outliers (92 outliers detected)
- Prints quality report summary

#### 5. Integration
- Merges datasets:
  - `merge_neighbourhoods`: Adds neighbourhood_group to listings
  - `merge_reviews`: Adds review counts and average review lengths
- Prints row/column counts after each merge

**Integrated Dataset:** (10,669, 83)

*Note: Some columns like neighbourhood_group initially had all NaNs and were filled as 'Unknown'*

#### 6. Aggregation
- Aggregates listings by category (e.g., neighbourhood_cleansed) using `aggregate_listings`
- Computes mean price, count, and average rating per group

**Top 10 Neighbourhoods by Average Price:**

| Neighbourhood | Avg Price | Listings Count | Avg Review Score |
|---------------|-----------|----------------|------------------|
| Arenella | 155.81 | 190 | 4.83 |
| Avvocata | 126.49 | 642 | 4.75 |
| Bagnoli | 87.41 | 63 | 4.68 |
| Barra | 64.82 | 17 | 4.75 |
| Chiaia | 184.90 | 729 | 4.75 |
| Chiaiano | 710.94 | 16 | 4.68 |
| Fuorigrotta | 227.49 | 207 | 4.81 |
| Mercato | 101.28 | 192 | 4.69 |
| Miano | 90.57 | 7 | 4.56 |
| Montecalvario | 270.36 | 721 | 4.75 |

#### 7. Sampling
- Samples 10% of listings for faster processing
- Stratified sampling by neighbourhood if available
- Falls back to random sampling if needed

**Sampled Dataset:** 1,068 rows

#### 8. Cleaning
- Uses `clean_data` to perform:
  - Basic cleaning
  - Removing or correcting invalid values
  - Column-specific adjustments
  - Duplicate removal (5 duplicate review rows removed)
- Prints updated shape after cleaning

**After Cleaning:** (1,068, 83)

#### 9. Handling Missing Values
- Identifies missing values per column (`identify_missing`)
- Imputes or handles missing values (`handle_missing_values`)
- Compares missing values before and after imputation

**Missing Values:**
- Before imputation: 11,103
- After imputation: 4,206

#### 10. Advanced Preprocessing

Uses the `AdvancedPreprocessor` class for comprehensive feature engineering:

##### 10.1 Derived Properties
Creates 8 new features:
- price_per_person
- host_experience_years
- availability_ratio
- amenity_count
- bathroom_bedroom_ratio
- avg_review_score
- location_desirability
- is_luxury

**Output:** (1,068, 91)

##### 10.2 Discretization & Binarization
- Discretized: price, reviews, availability, accommodates
- Binarized: expensive, many_reviews, superhost, instant_bookable

**Output:** (1,068, 99)

##### 10.3 Property Subset Selection
Filtered high-value listings (price ≥ $117)

**Output:** (270, 99)

##### 10.4 Data Transformations
- Applied scaling, log/sqrt transformations, one-hot encoding

**Output:** (270, 182)

##### 10.5 Dimension Reduction
- **PCA:** Reduced 114 numeric features → 27 components (95% variance)
- **Univariate Selection:** Kept 20 best features

**Output:**
- PCA dataset: (270, 27)
- Feature-selected dataset: (270, 20)

**Preprocessing Summary:**
- Original dataset: 1,068 rows, 83 columns
- Processed dataset: 270 rows, 182 columns
- Features added: 99

#### 11. Exploratory Data Analysis (EDA)

Uses the `EDAAnalyzer` class to perform comprehensive data exploration:

##### 11.1 Summary Statistics
- Numerical summary (mean, median, std, skewness, kurtosis) for up to 30 numeric columns
- Categorical summary (value counts per category) for up to 20 categorical columns

##### 11.2 Multivariate Analysis
- **Correlation Matrix:** Computes pairwise correlations for up to 25 numeric columns
- **Top Correlations:** Identifies strongest positive and negative correlations
- **PCA Analysis:** Reduces dimensionality and reports explained variance
- **Pairplots:** Generates scatter matrix for key features (price, ratings, reviews)
- **Grouped Summary:** Aggregates price by neighbourhood and room type

##### 11.3 Visualizations
All visualizations are saved to `eda_plots/` directory:
- `distribution_{column}.png` - Histogram with KDE for numeric columns
- `boxplot_{column}.png` - Boxplot for outlier visualization
- `correlation_heatmap.png` - Correlation matrix heatmap
- `pca_plot.png` - PCA components scatter plot
- `pairplot.png` - Pairwise relationships for key features

##### 11.4 Output Files
- `numerical_summary.csv` - Complete numerical statistics
- `correlation_matrix.csv` - Full correlation matrix
- `pca_components.csv` - PCA transformed data
- `grouped_summary_neighbourhood_price.csv` - Price statistics by neighbourhood

#### 12. Outlier Detection

Uses the `OutlierDetector` class to identify and validate outliers using multiple methods:

##### 12.1 Detection Methods

1. **IQR (Interquartile Range)**
   - Univariate method for individual columns
   - Threshold: Q1 - 1.5×IQR to Q3 + 1.5×IQR

2. **Z-Score**
   - Statistical method based on standard deviations
   - Threshold: |Z-score| > 3

3. **Isolation Forest**
   - Multivariate ensemble method
   - Contamination rate: 5%

4. **Local Outlier Factor (LOF)**
   - Density-based multivariate method
   - Contamination rate: 5%

5. **Mahalanobis Distance**
   - Distance-based method using PCA components
   - Threshold: 3.5

##### 12.2 Combined Outlier Score
- Computes overall score by summing all outlier flags
- Maps to types: normal (0), mild (1), strong (2), extreme (≥3)

##### 12.3 Validation & Filtering
- **Outlier Validation:** Requires agreement from ≥2 methods to confirm an outlier
- **False Positive Filtering:** Automatically filters out false detections
- **False Positive Rate:** Reports the percentage of false positives detected
- **Method Contributions:** Shows which methods contributed to false positives

**Example Results:**
- Total detected outliers: 159
- False positives filtered: 56 (35.2%)
- Confirmed outliers: 103
- Validation rate: 64.8%

##### 12.4 Output Columns

The final dataset includes comprehensive outlier detection columns:

- `outlier_iqr_{column}` - IQR outlier flags for each column
- `outlier_zscore_{column}` - Z-Score outlier flags for each column
- `outlier_iforest` - Isolation Forest outlier flag
- `outlier_lof` - LOF outlier flag
- `outlier_mahalanobis` - Mahalanobis Distance outlier flag (if available)
- `outlier_score` - Combined score (0 = normal, 1 = mild, 2 = strong, ≥3 = extreme)
- `outlier_type` - Classification: normal, mild, strong, extreme
- `outlier_validated` - Validated outliers (≥2 methods agree)
- `outlier_confirmed` - Confirmed outliers after filtering
- `outlier_false_positive` - False positive flags
- `outlier_agreement_count` - Number of methods that flagged each record

**After Outlier Detection:** (270, 225)

#### 13. Save Processed Data

Saves the **final dataset** to the processed dataset folder:
- `integrated_processed_listings.csv` (270 rows, 252 columns)

**Output includes:**
- All preprocessed features
- All outlier detection flags and scores
- Validation and confirmation columns
- False positive indicators

---

## Pipeline Key Features

### Modular Design
- All steps are **logged with timing** to track performance
- Errors in any step are **handled gracefully** with informative messages
- Designed to handle **large datasets efficiently** using sampling when necessary
- Modular design allows for **easy extension** of preprocessing or feature engineering steps

### Feature Engineering
- **99 new features** created through advanced preprocessing
- Derived properties capture domain-specific insights
- Discretization and binarization enable categorical analysis
- Transformations improve model compatibility

### Data Quality
- Comprehensive quality assessment at multiple stages
- Systematic handling of missing values, duplicates, and inconsistencies
- Outlier detection with multiple validation methods
- False positive filtering ensures data reliability

### Analysis Ready
- **252 total columns** in final dataset (includes 38 outlier detection columns)
- Complete EDA with visualizations and statistical summaries
- Ready for downstream analysis or machine learning modeling
- Comprehensive documentation of all transformations

---

## Output Files Summary

### Processed Data
- `integrated_processed_listings.csv` - Final processed dataset with all features and outlier flags

### EDA Outputs
- `eda_plots/` - Directory containing all visualizations
- `numerical_summary.csv` - Descriptive statistics
- `correlation_matrix.csv` - Correlation analysis
- `pca_components.csv` - Dimensionality reduction results
- `grouped_summary_neighbourhood_price.csv` - Neighborhood analysis

### Visualizations
- Distribution plots with KDE
- Boxplots for outlier visualization
- Correlation heatmaps
- PCA component plots
- Pairplots for feature relationships

---

## Notes and Highlights

- **Total columns after full preprocessing:** 252 (includes 38 outlier detection columns)
- **Total rows after high-value filtering:** 270
- Advanced preprocessing pipeline completed successfully
- EDA generates visualizations and summary statistics for data understanding
- Outlier detection identifies and validates outliers using 5 different methods
- False positive filtering reduces noise by requiring agreement from multiple methods
- Merge warnings handled by filling missing values with defaults
- All transformations are documented and reproducible

---

## Future Enhancements

- Add more sophisticated feature engineering techniques
- Implement additional outlier detection methods
- Create interactive visualizations using Plotly or Bokeh
- Add time-series analysis for review trends
- Implement geospatial analysis using latitude/longitude
- Create predictive models for pricing and demand

---

## License

This project is part of an academic course at the University of Prishtina and is intended for educational purposes.
