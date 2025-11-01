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
