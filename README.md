# Dataset Shift
An analyze of dataset shift in deep learning using "transfer learning" . We used Resnet50 and Densenet121 model. 

The dataset used in this study, 
1. Montgomery -available at https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html
2. Shenzhen -available at  https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html
3. TB Portals Program data maintained by the National Institute of Allergy and Infectious Diseases - available at https://tbportals.niaid.nih.gov/
4. Indian -Due to specific institutional requirements governing privacy protection, the Indian dataset were not available in public. 


Calculating Confidence Interval for AUC
========================================
![plot](./src/ci_auc.png)

Calculating Confidence Interval for Sensitivity and Scpecificity
================================================================

Two-sided confidence, Wilson method for Upper limit
----------------------------------------------------
![plot](./src/binomial_upper_limit_two_sided_wilson.png)

Two-sided confidence, Wilson method for Lower limit
----------------------------------------------------
![plot](./src/binomial_lower_limit_two_sided_wilson.png)
