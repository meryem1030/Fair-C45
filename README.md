# Fair-C45
1. This repository is the implementation of Multiple Fairness Criteria in Decision Tree Learning.
2. Each folder, which has been named by the name of dataset, has codes and dataset.
3. In each folder, name of each algorithm has been added at the end of .py file. 
4. Note: Results can be little different for Lexicographic and GRXFR models because of random attribute choose.
For example;
ADULT_RACE_LEXICOGRAPHIC.py: the code is Lexicographic algorithm, sensitive attribute is 'Race', and run on Adult dataset.
ADULT_RACE_C4.5: the code has base C4.5 algorithm, will run on Adult dataset. Sensistive attribute is 'Race'.
# Datasets:
1. ADULT: [1],[2],[3],[4],[5]
2. GERMAN: [1],[3],[6]
3. PROPUBLICA: [3],[7]
4. PROPUBLICA VIOLENT:[3],[7]
5. RICCI: [3],[6]
6. WINE : [8]
7. STUDENT MATH PERFORMANCE: [2],[9]
8. STUDENT PORTUGUESE PERFORMANCE: [2],[9]
9. DRUG : [10],[11]
10. BANK: [12],[13],[14]
11. SQF: [15],[16],[17]
12. LAW SCHOOL: [18],[19]
13. DUTCH: [20]
14. UFRGS: [21]

# To run the codes, 
1. As a first way, download .py files that start with same name and the relevant dataset in the same folder. For example, download .py files and dataset in ADULT file to same folder in your computer. Then, load the .py file on any text editor. To run the code, firstly open command promt on the folder path and run following line on the promt.
```python ADULT_AGE_C45.py``` and ```python ADULT_GENDER_LEXICOGRAPHIC_C45.py```
'''

repeat the process for all codes. Please note that the codes that has been named with their dataset and relevant dataset should be in same folder. Similar way how I created here.

2. Second way is the using Jyputer notebook in ANACONDA. Just copy and paste the all lines of code on notebook, then run.

# REFERENCES
[1] M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and S. Venkatasubramanian, “Certifying and Removing Disparate Impact *,” DOI: 10 . 1145 / 2766XXX .XXXXXXX. [Online]. Available: http : / / dx . doi . org /10.1145/2766XXX.XXXXXXX.

[2] M. Kearns, S. Neel, A. Roth, and Z. S. Wu, “An Empirical Study of Rich Subgroup Fairness for Machine Learning,” 2019. DOI: 10.1145/3287560.3287592.

[3] A. Valdivia, J. Sanchez-Monedero, and J. Casillas, ´“How fair can we go in machine learning? Assessing the boundaries of fairness in decision trees,” 2020.

[4]  Iosifidis, A. Roy, and E. Ntoutsi, “Parity-based cumulative fairness-aware boosting,” Knowledge and Information Systems, vol. 64, pp. 2737–2770, 2022. DOI:10. 1007 / s10115 - 022 - 01723 - 3. [Online]. Available:https://doi.org/10.1007/s10115-022-01723-3.

[5] G. Maheshwari and M. Perrot, “FairGrad: Fairness Aware Gradient Descent,”

[6] S. A. Friedler, C. Scheidegger, S. Venkatasubramanian, et al., “A comparative study of fairness-enhancing interventions in machine learning * ACM Reference Format,” 2019. DOI: 10.1145/3287560.3287589. [Online].Available: https://doi.org/10.1145/3287560.3287589.

[7] T. Le Quy, A. Roy, V. Iosifidis, W. Zhang, and E.Ntoutsi, “A survey on datasets for fairness-aware machine learning,” Wiley Interdisciplinary Reviews: Data
Mining and Knowledge Discovery, vol. 12, no. 3, e1452,May 2022, ISSN: 1942-4795. DOI: 10 . 1002 / WIDM .1452. [Online]. Available: https://onlinelibrary.wiley.
com / doi / full / 10 . 1002 / widm . 1452 % 20https : / /onlinelibrary.wiley.com/doi/abs/10.1002/widm.1452%20https://wires.onlinelibrary.wiley.com/doi/10.1002/
widm.1452.

[8] Cortez, A. Cerdeira, F. Almeida, T. Matos, and J.Reis, “Modeling wine preferences by data mining from physicochemical properties,” Decision Support Systems,
vol. 47, no. 4, pp. 547–553, Nov. 20

[9] P. Cortez and A. Silva, “USING DATA MINING TO PREDICT SECONDARY SCHOOL STUDENT PERFORMANCE,”

[10] . Fehrman, V. Egan, A. N. Gorban, J. Levesley, E. M. Mirkes, and A. K. Muhammad, “Personality Traits and Drug Consumption,” Personality Traits and Drug
Consumption, 2019. DOI: 10.1007/978-3-030-10442-9.

[11] H. Do, P. Putzel, A. Martin, P. Smyth, and J. Zhong, “Fair Generalized Linear Models with a Convex Penalty,” 2022. [Online]. Available: https://github.com/
hyungrok-do/fair-gl.

[12] A. P. Barata, F. W. Takes, H. J. v. d. Herik, and C. J.Veenman, “Fair Tree Classifier using Strong Demographic Parity,” Oct. 2021. DOI: 10.48550/arxiv.2110.
09295. [Online]. Available: https://arxiv.org/abs/2110.09295v3.

[13] V. Grari, B. Ruf, S. Lamprier, and M. Detyniecki, “Fair Adversarial Gradient Tree Boosting,” 2019.

[14]  Ravichandran, D. Khurana, A. Labs, et al., “FairXGBoost: Fairness-aware Classification in XGBoost,”2020. DOI: 10 . 1145 / 1122445 . 1122456. [Online].
Available: https://doi.org/10.1145/1122445.1122456.

[15] S. Goel, J. M. Rao, and R. Shroff, “Precinct or prejudice? Understanding racial disparities in New York City’s stop-and-frisk policy,” https://doi.org/10.1214/15-AOAS897, vol. 10, no. 1, pp. 365–394, Mar. 2016, ISSN: 1932-6157. DOI:10 . 1214 / 15 - AOAS897. [Online]. Available: https :/ / projecteuclid . org / journals / annals - of - applied -statistics / volume - 10 / issue - 1 / Precinct - or- prejudice -Understanding - racial - disparities - in - New - York -Citys / 10 . 1214 / 15 - AOAS897 . full % 20https :/ / projecteuclid . org / journals / annals - of - applied -statistics / volume - 10 / issue - 1 / Precinct - or- prejudice -Understanding - racial - disparities - in - New - York -Citys/10.1214/15-AOAS897.short

[16] J. MacDonald and A. A. Braga, “Did Post-Floyd et al. Reforms Reduce Racial Disparities in NYPD Stop,Question, and Frisk Practices? An Exploratory Analysis Using External and Internal Benchmarks,” Justice Quarterly, vol. 36, no. 5, pp. 954–983, Jul. 2019, ISSN:17459109. DOI: 10 . 1080 / 07418825 . 2018 . 1427278.
[Online]. Available: https://doi.org/10.1080/07418825.2018.1427278.

[17] https://www.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page

[18]  Y. Bechavod and K. Ligett, “Penalizing Unfairness in Binary Classification,” Jun. 2017. [Online]. Available:https://arxiv.org/abs/1707.00044v3.

[19] A. Ruoss, M. Balunovic, M. Fischer, and M. Vechev, “Learning Certified Individually Fair Representations,” Advances in Neural Information Processing Systems,
vol. 2020-December, Feb. 2020, ISSN: 10495258. [Online]. Available: https://arxiv.org/abs/2002.10312v

[20] F. Kamiran, T. Calders, and M. Pechenizkiy, “Discrimination aware decision tree learning,” Proceedings -IEEE International Conference on Data Mining, ICDM,
pp. 869–874, 2010, ISSN: 15504786. DOI: 10 . 1109 /ICDM.2010.50.

[21] J. Zhang, I. Beschastnikh, S. Mechtaev, and A. Roychoudhury, “Fair Decision Making via Automated Repair of Decision Trees,” Proceedings - International
Workshop on Equitable Data and Technology, FairWare 2022, pp. 9–16, 2022. DOI: 10.1145/3524491.3527306.[Online]. Available: https://dl.acm.org/doi/10.1145/
3524491.3527306.



