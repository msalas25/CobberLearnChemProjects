In the Alkane Boiling Point (chapter 3) folder, the purpose of this activity is to use Python and Matplotlib to create a scatterplot showing how the boiling point of the first ten straight-chain alkanes changes as the number of carbon atoms increases. The activity also helps practice using chemical data, creating graphs, and checking whether the observed trend makes scientific sense.

In PubChemFetcher (Chapter 5), the purpose of this activity is to use Python and the PubChemPy library to retrieve chemical information directly from the PubChem database. This includes properties such as molecular weight, molecular formula, and SMILES strings. This helps us understand how to access, extract, and organize chemical data using Python without manually searching through databases.

In Molecule Explorer (Chapter 6) the purpose of this activity is to use Python and RDKit to analyze molecular structures from SMILES strings and calculate chemical properties such as molecular weight, hydrogen bond donors, and TPSA. This helps us understand how molecular structures relate to their chemical properties without relying on pre-built datasets.

In MakingDataWhole (Chapter 7), the purpose of this activity is to use Python and machine learning techniques such as mean imputation, KNN, and Random Forest to predict missing age values in the Titanic dataset. This helps us understand how different imputation methods handle missing data and how to evaluate their accuracy using Mean Absolute Error (MAE).

In the Error Metrics (Chapter 8) folder, the purpose of this activity is to evaluate how accurately a model makes predictions by comparing the actual and predicted values. The error metrics and graphs help show how well the model performs and identify where the largest prediction errors occur.

In DecisionTreeClassifier (Chapter 9), the purpose of this activity is to use Python, pandas, and scikit-learn to build a decision tree model that predicts whether a molecule is water-soluble based on its molecular weight, hydrogen bond donors, and hydrogen bond acceptors. This helps us understand how machine learning uses chemical properties to classify molecules and visualize the decision-making process.

In ElementClustering (Chapter 10), the purpose of this activity is to use Python, pandas, Matplotlib, and K-Means clustering to analyze periodic trends among alkali metals. By comparing atomic radius and first ionization energy, we can visualize relationships between elements and group them based on similarities in their chemical properties. This helps us understand how machine learning can identify patterns in the periodic table without predefined classifications.

In the Gradient Descent (Chapter 11) folder, the purpose of this activity is to use Python, NumPy, and scikit-learn to fit a linear regression model to noisy data. The model predicts the slope and intercept, while Mean Squared Error (MSE) measures how accurately the line fits the data. The best-fit line achieved an MSE of 3.3006, and the loss landscape helps visualize how different slope and intercept values affect prediction error.

In the Weasel Program (Chapter 12) folder, the purpose of this activity is to use Python to demonstrate how evolutionary algorithms work through mutation and selection. The program starts with a random string and gradually evolves it into the target phrase by selecting the best offspring from each generation. The final phrase was successfully achieved in 84 generations with a perfect fitness score of 28/28. The fitness graph shows how the program improves over time.

In the Artificial Neural Network (Chapter 13) folder, the purpose of this activity is to use Python, scikit-learn, and Matplotlib to compare linear regression and neural networks for predicting boiling points from molecular weight. The models were evaluated using MAE, MSE, and R², and their predictions were compared using graphs. The neural network performed slightly better on the training data, while linear regression predicted the unseen Butane sample more accurately. Different neural network architectures were also tested to understand how model complexity affects performance.

In the Q-Learning (Chapter 15) folder, the purpose of this activity is to use Python and Gymnasium to train an AI agent to navigate the FrozenLake environment through trial and error. The agent uses a Q-table to learn which actions lead to the best rewards and improve its decision-making over time. After 100,000 training episodes, the agent achieved a 72% success rate, demonstrating how reinforcement learning allows AI to learn from experience.

In the California Housing Regression (Chapter 16) folder, the purpose of this activity is to use Python and scikit-learn to predict housing prices using different regression models. The activity also uses graphs and R² scores to compare model performance, identify important housing features, and understand how factors such as income, house age, and location affect predicted housing prices.

















I downloaded the ToxCast/Tox21 dataset and started looking at data related to estrogen receptors.
I used Python to separate the data into three files: ERα, ERβ, and ERβ viability. I also checked whether the files had missing information or duplicate rows.One problem I ran into was figuring out how the database connects the test results to the actual chemical names. I am still working on that part.My next step is to identify which chemicals were tested and understand how to compare the results between the estrogen receptors and the cell viability data. (estrogen agonist folder) 


Descriptive statistics were calculated for the TOX21_ERa_BLA_Agonist_ratio dataset (AEID 785), which contained 10,499 assay records. The dataset had no duplicate rows or missing values in the analyzed columns. Using a provisional activity threshold of 0.90, 592 records (5.64%) were classified as active, 9,905 (94.34%) as inactive, and 2 (0.02%) as unable to fit. The most frequent model type was poly1 (4,144 records), followed by exp4 (2,576 records). These results describe the distribution of assay records; the activity labels and number of unique chemicals require further validation. 


Under the Data folder(ERa estradol similarity graph) I used Morgan fingerprints and Tanimoto similarity to compare estradiol with 7,826 other chemicals in the ERα BLA assay dataset. The closest matches included estriol, estrone, and several estradiol derivatives. All 10 top matches were provisionally classified as active. The analysis also showed that fingerprints without chirality cannot distinguish certain stereoisomers.


For the ERa estrogen solubility graph, it compares the water solubility of estradiol, estriol, and estrone using reported experimental data from PubChem. 

For ERa logP Vs. solubility graph compares LogP and water solubility in three estrogens to explore how lipophilicity relates to water solubility.

For ERa descriptor activity graph compares average molecular weight, LogP, and TPSA between active and inactive chemicals in the ERα assay dataset.

For ERa descriptor Box plots comparing molecular weight, LogP, and TPSA distributions between active and inactive ERα assay chemicals.

 For ERa similarities vs activity graph shows the percentage of active ERα assay chemicals across different structural similarity ranges to estradiol. 

 For Era similarity group counts graph shows how many chemicals fall into each structural similarity range compared with estradiol.
