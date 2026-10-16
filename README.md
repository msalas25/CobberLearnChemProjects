AlkaneBoilingPoints This simple code plots the number of car-
bon atoms in the alkane on the x-axis and the boiling points of

that alkane on the y-axis. It generates a properly labeled graph
using Matplotlib.
This is part of the CobberLearnChem Machine Learning course
through Concordia College.

Alongside my public coding projects, I’m keeping a private
ethics portfolio where I reflect on what I’m learning and how
it’s shaping the kind of scientist I want to become.


In the Alkane Boiling Point (chapter 3) folder, the purpose of this activity is to use Python and Matplotlib to create a scatterplot showing how the boiling point of the first ten straight-chain alkanes changes as the number of carbon atoms increases. The activity also helps practice using chemical data, creating graphs, and checking whether the observed trend makes scientific sense.


In Molecule Explorer (Chapter 6) the purpose of this activity is to use Python and RDKit to analyze molecular structures from SMILES strings and calculate chemical properties such as molecular weight, hydrogen bond donors, and TPSA. This helps us understand how molecular structures relate to their chemical properties without relying on pre-built datasets.


In the Error Metrics (Chapter 8) folder, the purpose of this activity is to evaluate how accurately a model makes predictions by comparing the actual and predicted values. The error metrics and graphs help show how well the model performs and identify where the largest prediction errors occur.


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
