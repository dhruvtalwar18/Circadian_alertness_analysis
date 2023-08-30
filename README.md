
# Circadian-Rhythm-Driven-Alertness-Investigation

**Motivation:** 

The motivation behind circadian rhythm is to regulate the timing and quality of sleep, which is crucial for overall health and wellbeing. Any disruptions to this internal biological clock can lead to difficulty sleeping, negatively impacting alertness, productivity, and increasing healthcare utilization and economic costs. 

**Data Analysis:**

A Study based on how alertness is affected by different factors in life, including sleep, food and exercise. We demonstarted that  how effectively an individual awakens in the hours following sleep is not associated with their genetics, but instead, four independent factors: sleep quantity/quality the night before, physical activity the day prior, a breakfast rich in carbohydrate, and a lower blood glucose response following breakfast. Furthermore, an individual’s set-point of daily alertness is related to the quality of their sleep, their positive emotional state, and their age.

We performed analysis stated below:

1. Demographics Analysis
2. Sleep Analysis
3. Heritability Analysis
4. Physical Activity Analysis
5. Food Analysis

*Dataset: [Alertness data](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-34503-2/MediaObjects/41467_2022_34503_MOESM3_ESM.xlsx)*

**“PREDICT1”**  is a two-country (UK, US) longitudinal study. The data is being collected from the user through a mobile app called Zoe App. 



## **File Structure**

```

├── Data
│   ├──processed_data.csv   <- Processed dataset
│   ├──raw_data1.xlsx
│   └──raw_data2.xlsx       <- Raw dataset
│   
│                     
├── model             <- best model saved in pickle file   
│   └──xgb_reg.pkl 
│
│
├── plots                          <- plots created using bokeh, plotly, matplotlib, and seaborn
│   ├──demographics          
│   ├──descriptive_statistics     
│   ├──food
│   ├──heritability
│   ├──mode_prediction
│   ├──physical_activity
│   ├──sleep
│   └──Visualization.pdf
│
├── res  
│   └──predicted_result.csv     <- predicted model results
│
│
├── src                             <- src files to create the plots and train models
│   ├──all_visualization.ipynb      <- Contains visualization of all the analysis in a single .ipynb file
│   ├──demographics_sleep.ipynb     
│   ├──descriptive_stats.ipynb
│   ├──exercise_food.ipynb
│   ├──heritability.ipynb
│   ├──model.ipynb
│   └──model.py                 <- Contains model processing, transformation, normalization, training, and testing  in a single .py file 
│   
│
├── ECE_143_Final.pdf           <- Final Presentation
│
├── readme.md                   <- README
│
└── requirement.yaml            <- all required packages 
```

---
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Used Modules

#### Data Processing
1. NumPy
2. Pandas
3. Scipy

#### Data Visualization
1. Bokeh
2. Plotly
3. Matplotlib
4. Seaborn

#### Data Modeling
1. Scikit-Learn
2. XgBoost
3. LightGBM
4. SHAP


### Prerequisites

What things you need to install the software and how to install them

```
conda env create -f requirement.yaml

```


## How to run


1. Clone the repository to the local system.
    ```
    git clone https://github.com/amitashnanda/Circadian-Rhythm-driven-Alertness-Investigation.git

    ```
2. Open ```/data/```

    ```/raw_data1.xlsx``` is used for demographics, sleep, food, heritability, and physical activities analysis.

    ```/raw_data2.xlsx``` is used for descriptive statistics analysis
    
    ```/processed_data.csv``` is the processed data 

3. Open ```/src/``` 

4. To run visualization for all the analysis. Inside the conda environment created using ```requirement.yaml```.

    ```
    jupyter notebook  all_visualization.ipynb

    ```
5. To run individual other files 

    ```
    jupyter notebook <file_name>

    ```
6. To run the model  analysis, transformation, normalization, training, and prediction.

    ```
    python3 model.py

    ```

## Conclusion

1. Individuals who wake up early have higher alertness levels despite the same amount of good sleep as those who wake up late. This finding supports the concept of sleep efficiency and circadian rhythm, emphasizing the importance of waking up early for optimal alertness.

2. Increased wakefulness after sleep onset leads to reduced alertness the following day.

3. The macronutrient composition of breakfast significantly affects alertness after sleep, with high-carbohydrate breakfast increasing alertness and high-protein breakfast decreasing alertness.

4. The impact of breakfast on alertness is not solely dependent on energy content or postprandial blood glucose levels, but also on the specific macronutrient composition.

5. Higher physical activity levels on the previous day (represented by higher M10 values) are associated with increased next-day morning alertness, while greater movement during the least active period (represented by higher L5 values) leads to lower alertness.


## **Sources**

*[Vallat, R., Berry, S.E., Tsereteli, N. et al. How people wake up is associated with previous night’s sleep together with physical activity and food intake. Nat Commun 13, 7116 (2022)](https://doi.org/10.1038/s41467-022-34503-2)*



