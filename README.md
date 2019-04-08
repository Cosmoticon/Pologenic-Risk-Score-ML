# ML
Guide to calucalte polygenic risk score using Imputed Dosage genetic data: It's a binary classification probelm where X- Genetic Data and Y- Class label (for example 1 for an invidual witha disease; 0 not)

Step 1: Input Data
  1. Input: Traing Data (X and Y)
  2. Input: Test Data (X and Y)

Step 2: Polygenic Risk Generation
  Generate weights associated with each input variable based on the relationship between input variables and output:
  Two Aprroaches
    1. Penalized Logistic Regression
    2. ELastic Net

Step 3: Validation with Test Data
    1. ROC calculation
  
