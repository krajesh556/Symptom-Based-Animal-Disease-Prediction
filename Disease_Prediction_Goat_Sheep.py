from fontTools.config import OPTIONS
from mpl_toolkits.mplot3d import Axes3D
from numpy import interp
from scipy.ndimage import label
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tkinter import *
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tkinter import messagebox
import shap
from tqdm import tk
import os
from tkinter import Toplevel, Label, PhotoImage
import missingno as msno
import seaborn as sns
from  sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, plot_roc_curve
from sklearn.decomposition import PCA
symptom_options = [
    'Fever', 'Reduced_Milk_Production', 'Weight_Loss', 'Pyrexia',
    'Discharge_Eyes_Nose', 'Swelling of the lips, tongue, and head', 'Lameness and Oral Lesions',
    'Lethargy_snotty nose', 'Foul_Smelling_diarhoea', 'Reddening around the coronary band', 'Lesions_on_mouth', 'loss of appetite_High Fever',
    'Skin_Lesions', 'Early delivery or Abortions', 'Red spots', 'Initial_Blisters', 'Modified milk texture', 'Misshapen udders',
    'Weak_Lambs'
]
symptom_options1 = [
'Pyrexia',
    'Discharge_Eyes_Nose', 'Swelling of the lips, tongue, and head', 'Lameness and Oral Lesions',
    'Lethargy_snotty nose', 'Foul_Smelling_diarhoea', 'Reddening around the coronary band', 'Lesions_on_mouth', 'loss of appetite_High Fever',
    'Skin_Lesions', 'Early delivery or Abortions', 'Red spots', 'Initial_Blisters', 'Modified milk texture', 'Misshapen udders',
    'Weak_Lambs'
]
common_symptoms_options = [
    'Fever', 'Reduced_Milk_Production', 'Weight_Loss'
]
disease_explanations = {
'Foot-Mouth-Disease(Goat)':"Agent: Foot and mouth disease virus (FMDV), a member of the Picornaviridae family.\n"
                               "\nClinical signs: Clinical signs of foot and mouth disease (FMD) in goats include fever, excessive salivation, lameness, and vesicle formation on the oral mucosa and coronary bands.\n"
                               "\nTransmission: Transmission of FMD in goats occurs via direct contact with infected animals, contaminated equipment, aerosols, and biting insects.\n"
    "\nPrevention and control: Preventing and controlling FMD in goats involves vaccination, implementing biosecurity measures, quarantine procedures, vector control, and maintaining hygiene protocols.\n",
'Peste des petits ruminants(Goat)':"Agent: Peste des petits ruminants virus (PPRV), a member of the Paramyxoviridae family.\n"
    "\nClinical signs: Clinical signs of Peste des petits ruminants (PPR) in goats include fever, nasal discharge, ocular discharge, diarrhea, pneumonia, and sometimes death.\n"
    "\nTransmission: Transmission of PPR in goats occurs through direct contact with infected animals, respiratory secretions, feces, and contaminated environments.\n"
    "\nPrevention and control: Preventing and controlling Peste des petits ruminants (PPR) in goats involves vaccination, strict quarantine measures, biosecurity protocols, and proper sanitation practices.\n",
'BlueTongue(Goat)':"Agent: Blue Tongue Virus (BTV), a member of the Orbivirus genus.\n"
    "\nClinical signs: Clinical signs of Blue Tongue disease in goats include fever, nasal discharge, swollen face and lips, congestion of oral and nasal mucosa, ulcerations on the tongue and mouth, lameness, and sometimes death.\n"
    "\nTransmission: Blue Tongue Virus (BTV) in goats is primarily transmitted by biting midges of the Culicoides species. It can also spread through direct contact with infected animals or through the placenta from an infected dam to her fetus.\n"
    "\nPrevention and control: Preventing and controlling Blue Tongue disease in goats involves vaccination, vector control (e.g., reducing the population of Culicoides midges), quarantine measures, and avoiding the introduction of infected animals into susceptible herds.\n",
'Rift Valley fever(Goat)':"Agent: Rift Valley Fever Virus (RVFV), a member of the Phlebovirus genus.\n"
    "\nClinical signs: Clinical signs of Rift Valley Fever in goats include high fever, lethargy, weakness, nasal discharge, jaundice, abortion in pregnant does, and sometimes death. In severe cases, neurological signs such as tremors and convulsions may also be observed.\n"
    "\nTransmission: Rift Valley Fever Virus (RVFV) is primarily transmitted to goats through the bites of infected mosquitoes, primarily species of the Aedes and Culex genera. It can also spread through direct contact with infected tissues, blood, or body fluids from infected animals.\n"
    "\nPrevention and control: Preventing and controlling Rift Valley Fever in goats involves vaccination, vector control (e.g., using insecticides and mosquito nets), avoiding contact with infected animals or their products, and practicing good hygiene and biosecurity measures on farms. It's also essential to monitor and report any suspected cases to veterinary authorities for prompt intervention.\n",
'Border Disease(Goat)':"Agent: Border Disease Virus (BDV), a member of the Pestivirus genus.\n"
    "\nClinical signs: Clinical signs of Border Disease in goats include reproductive problems such as abortion,early births, weak or deformed kids, and the birth of small-sized kids. Infected goats may also exhibit signs of ill-thrift, including poor growth, lethargy, and a rough hair coat. Border Disease can also lead to immunosuppression, making infected goats more susceptible to other diseases.\n"
    "\nTransmission: Border Disease Virus (BDV) can be transmitted vertically from an infected doe to her offspring during pregnancy. Horizontal transmission can occur through direct contact with infected animals or their bodily fluids, as well as through contaminated environments or fomites.\n"
    "\nPrevention and control: Preventing and controlling Border Disease in goats involves implementing biosecurity measures, such as quarantine and testing of new animals before introduction to the herd, as well as maintaining a closed herd. Vaccination of susceptible animals, especially replacement breeding stock, can also help prevent the spread of the disease. Proper hygiene practices and disinfection of equipment and facilities are essential to minimize the risk of transmission.\n",
'GoatPox(Goat)':"Agent:The agent for Goat Pox is the Goatpox virus, which belongs to the Poxviridae family.\n"
    "\nClinical signs:Clinical signs of Goat Pox include fever, nasal discharge, coughing, and characteristic skin lesions.\n"
    "\nTransmission:Transmission of the Goatpox virus occurs through direct contact with infected animals or contaminated objects, as well as through insect vectors like mosquitoes and ticks.\n"
    "\nPrevention and control: Prevention and control measures for Goat Pox include vaccination, strict biosecurity protocols, and proper herd management practices.\n",
'ORF(Goat)':"Agent: Orf virus, a member of the Parapoxvirus genus.\n"
    "\nClinical signs: Orf in goats typically presents as localized, proliferative, and often painful lesions on the lips, oral mucosa, and occasionally on the udder, teats, or feet. These lesions may initially appear as papules or vesicles and progress to thick, scabby crusts. Affected goats may exhibit signs of discomfort, reluctance to nurse, and in severe cases, may experience weight loss or systemic illness.\n"
    "\nTransmission: Orf virus is primarily transmitted through direct contact with infected animals or contaminated fomites such as feeders, water troughs, or bedding material. Infection can also occur through contact with contaminated environmental surfaces or through ingestion of contaminated material. Mechanical transmission by biting insects may also play a role in the spread of the virus.\n"
    "\nPrevention and control: Prevention and control of Orf in goats involve implementing biosecurity measures to prevent introduction of the virus into herds, as well as management practices to minimize transmission within a herd. This includes isolation of affected animals, proper wound care, and disinfection of contaminated areas and equipment. Vaccination may also be considered in endemic areas or in herds with a history of recurrent outbreaks.\n",
'Mastitis(Goat)':"Agent: Mastitis is most commonly caused by bacterial pathogens, including Staphylococcus aureus, Streptococcus spp., Escherichia coli, and others.\n"
    "\nClinical signs: Mastitis in goats presents with inflammation of the mammary gland, leading to swelling, heat, redness, and pain in the affected udder. Affected goats may exhibit signs such as reluctance to nurse, reduced milk production, changes in milk color or consistency, and systemic signs of illness such as fever and lethargy.\n"
    "\nTransmission: Mastitis can occur through the introduction of infectious agents into the teat canal, typically during milking procedures or through environmental contamination of the udder. Factors such as poor hygiene, inadequate milking practices, and environmental stressors can increase the risk of mastitis.\n"
    "\nPrevention and control: Prevention and control of mastitis in goats involve implementing good management practices, including maintaining proper hygiene during milking, ensuring adequate nutrition and housing conditions, and promptly treating any cases of mastitis with appropriate antibiotics. Regular monitoring of udder health and maintaining a clean and dry environment can also help reduce the risk of mastitis.\n"
}
disease_options = ['Foot-Mouth-Disease(Goat)','Peste des petits ruminants(Goat)','BlueTongue(Goat)','Rift Valley fever(Goat)','Border Disease(Goat)',
                   'GoatPox(Goat)','ORF(Goat)','Mastitis(Goat)']
l2 = []
for i in range(0, len(symptom_options)):
    l2.append(0)
df = pd.read_csv(r'C:\Users\User\OneDrive\Desktop\python_programs\Data_sets_Cows_Buffalo_Goats\Goats.csv')
DF = pd.read_csv(r'C:\Users\User\OneDrive\Desktop\python_programs\Data_sets_Cows_Buffalo_Goats\Goats.csv', index_col='Disease')
df.replace({'Disease': {'Foot-Mouth-Disease(Goat)':0,'Peste des petits ruminants(Goat)':1,'BlueTongue(Goat)':2,'Rift Valley fever(Goat)':3,'Border Disease(Goat)':4,
                   'GoatPox(Goat)':5,'ORF(Goat)':6,'Mastitis(Goat)':7}}, inplace=True)
print("\n********** THE ATTRIBUTES THAT ARE PRESENT IN THE DATA SET **********")
print(df.head(10))
print("\n********** THE TOTAL NUMBER OF COLUMNS AND THE ROWS THAT ARE PRESENT IN THE DATA SET **********")
print(df.shape)
print("\n********** DISPLAYS THE INFORMATION ABOUT A DataFrameObject **********")
print(df.info())
print("\n********** SETS 3-DECIMAL FORMAT & SUMMARIZES TRANSPOSED NUMERICAL COLUMNS **********")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(df.describe().transpose())
print("\n********** THE TOTAL NUMBER OF COLUMNS AND THE ROWS THAT ARE PRESENT IN THE DATA SET **********")
print(df.shape)
print("\n********** FINDING THE MISSING VALUES **********")
print(df.isnull())
print("\n********** CALCULATES THE PERCENTAGES OF THE MISSING VALUES **********")
print(df.isnull().sum()/len(df)*100)
print("\n********** CREATING AN VISUALIZATION FOR THE MISSING DATA **********")
msno.bar(df,figsize=(6,2))
plt.show()
print("\n********** CHECKING AND REMOVING THE DUPLICATE ROWS FROM THE DATA FRAME **********")
print(f"\nTHE TOTAL NUMBER OF DUPLICATE VALUES IN THE DATA FRAME ARE: {df.duplicated().sum()}")
print("\n********** DISPLAYS THE INFORMATION ABOUT A DataFrameObject **********")
print(df.info())
print("\n********** DISPLAYS THE UNIQUE VALUES IN THE Age COLUMN **********")
print(df['Age'].unique())
print("\nDISPLAYING THE PIE CHART FOR THE AGE DISTRIBUTION")
age_counts = df['Age'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('AGE DISTRIBUTION AMONG THE SHEEPS AND GOATS')
plt.axis('equal')
plt.show()
print("\nDISPLAYING THE PIE CHART FOR THE BREED_NAME DISTRIBUTION.....")
breed_counts = df['Breed_name'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(breed_counts, labels=breed_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('BREED NAME DISTRIBUTION AMONG THE SHEEPS AND GOATS')
plt.axis('equal')
plt.show()
print("\nDISPLAYING THE PIE CHART FOR THE GENDER DISTRIBUTION.....")
gender_counts = df['Gender '].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('GENDER DISTRIBUTION AMONG THE SHEEPS AND THE GOATS')
plt.axis('equal')
plt.show()
print("\nGROUPING DATA BY Bree_name AND Gender, AND CALCULATING THE MEAN AGE FOR EACH GROUP.....")
grouped_data = df.groupby(['Breed_name', 'Gender '])['Age'].mean().unstack()
grouped_data.plot(kind='bar', figsize=(12, 8), color=['c', 'g'])
plt.title('Average Age by Breed Name and Gender')
plt.xlabel('Breed Name')
plt.ylabel('Average Age')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nCOUNT PLOTS FOR EACH SPECIFIED ATTRIBUTE WITH RESPECT TO THE PREDICTED DISEASE....")
attributes = ['Pyrexia',
    'Discharge_Eyes_Nose', 'Swelling of the lips, tongue, and head', 'Lameness and Oral Lesions',
    'Lethargy_snotty nose', 'Foul_Smelling_diarhoea', 'Reddening around the coronary band', 'Lesions_on_mouth', 'loss of appetite_High Fever',
    'Skin_Lesions', 'Early delivery or Abortions', 'Red spots', 'Initial_Blisters', 'Modified milk texture', 'Misshapen udders',
    'Weak_Lambs']
plt.figure(figsize=(15,10))
for i, attribute in enumerate(attributes):
    plt.subplot(4, 4, i + 1)
    sns.countplot(x=attribute, hue='Disease', data=df)
    plt.legend().remove()
    plt.title(attribute)
plt.legend(title='Disease', bbox_to_anchor=(1.05, 1.2), loc='upper center')
plt.tight_layout()
plt.show()
print("\nCALCULATES THE CORRELATION MATRIX OF THE DataFrame")
print(df.corr())
print("\nNOW WE CAN SEE THE CORRELATION MATRIX USING HEAT MAP")
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(df.corr(), ax=ax, center=0, annot=True)
plt.title("Correlation Matrix")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
print("\n LET'S MAKE MAKE AN BOX PLOT FOR THE ATTRIBUTES")
plt.figure(figsize=(20,10))
df.boxplot(grid=False)
plt.show()
X = df[symptom_options]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Disease'].astype(str))
noise_level = 0.3
X_noisy = X + np.random.normal(0, noise_level, X.shape)
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)
np.ravel(y)
def scatterplt():
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'cyan', 'magenta']
    for disease, color in zip(disease_options, colors):
        x = ((DF.loc[disease]).sum())
        x_numeric = pd.to_numeric(x, errors='coerce')
        x_filtered = x_numeric[x_numeric > 0]
        y = x_filtered.index
        plt.figure(figsize=(10, 6))
        plt.plot(y, x_filtered.values, marker='o', color=color, linestyle='-')
        plt.title(f"{disease} - Symptom Occurrences", color='blue')
        plt.xlabel('Symptoms', color='green')
        plt.ylabel('Occurrences', color='purple')
        plt.grid(True)
        for i, (symptom, occurrence) in enumerate(zip(y, x_filtered.values)):
            plt.text(symptom, occurrence, f"{symptom}: {occurrence}", ha='center', va='bottom', rotation=45,color='orange')
        plt.show()
scatterplt()
def scatterinp(sym1, sym2, sym3, sym4):
    x = [sym1, sym2, sym3, sym4]
    y = [0, 0, 0, 0]
    if sym1 != 'select here':
        y[0] = 1
    if sym2 != 'select here':
        y[1] = 1
    if sym3 != 'select here':
        y[2] = 1
    if sym4 != 'select here':
        y[3] = 1
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.show()
root = Tk()
pred1 = StringVar()
acc1 = StringVar()
pred2 = StringVar()
acc2 = StringVar()
pred3 = StringVar()
acc3 = StringVar()
pred4 = StringVar()
acc4 = StringVar()
def check_duplicate_symptoms(symptoms):
    return len(symptoms) == len(set(symptoms))
def Dt_classification_report(y_test,y_pred,target_names):
    report=classification_report(y_test,y_pred,target_names=target_names,output_dict=True)
    report_df=pd.DataFrame(report).transpose()
    report_df=report_df.round(2)
    print(report_df)
    plt.figure(figsize=(10,6))
    sns.heatmap(report_df.iloc[:-1,:].T,annot=True,cmap="Blues",fmt=".2f")
    plt.title('Classification Report--Decision Tree')
    plt.show()
def DecisionTree():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp = messagebox.askokcancel("System", "Kindly fill the Name")
        if comp:
            root.mainloop()
    elif ((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")):
        pred1.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill At least first two symptoms")
        if sym:
            root.mainloop()
    else:
        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]
        if psymptoms.count("select here") < 4 and not check_duplicate_symptoms(psymptoms):
            messagebox.showerror("Input Error", "Please select distinct symptoms.")
            return
        clf3 = tree.DecisionTreeClassifier()
        clf3 = clf3.fit(X_train, y_train)
        y_pred = clf3.predict(X_test)
        acc1.set(str(accuracy_score(y_test, y_pred)))
        scaler = StandardScaler()
        print("Decision Tree")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))
        print("Confusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        print("feature names:", list(X.columns))
        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]
        for k in range(0, len(symptom_options)):
            for z in psymptoms:
                if z == symptom_options[k]:
                    l2[k] = 1
        inputtest = [l2]
        predict = clf3.predict(inputtest)
        predicted = predict[0]
        h = 'no'
        for a in range(0, len(disease_options)):
            if predicted == a:
                h = 'yes'
                break
        if h == 'yes':
            pred1.set(" ")
            pred1.set(disease_options[a])
        else:
            pred1.set(" ")
            pred1.set("Not found")
            messagebox.showinfo("Prediction Result", "Disease not found")
        print("Selected Symptoms:", psymptoms)
        print("Test Set Predictions:", label_encoder.inverse_transform(y_pred))
        pred1.set(" ")
        pred1.set(disease_options[a])
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',xticklabels=disease_options, yticklabels=disease_options)
        plt.title('Confusion Matrix - Decision Tree')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        plt.figure(figsize=(12, 6))
        plot_tree(clf3, filled=True, feature_names=X.columns, class_names=label_encoder.classes_, fontsize=5)
        plt.title('Unpruned Decision Tree', fontsize=12)
        plt.show()
        path = clf3.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)
        train_scores = [clf.score(X_train, y_train) for clf in clfs]
        test_scores = [clf.score(X_test, y_test) for clf in clfs]
        plt.figure(figsize=(10, 6))
        plt.plot(ccp_alphas[:-1], train_scores[:-1], marker='o', label='train', drawstyle="steps-post")
        plt.plot(ccp_alphas[:-1], test_scores[:-1], marker='o', label='test', drawstyle="steps-post")
        plt.xlabel("Alpha")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Alpha for Decision Tree Pruning")
        plt.legend()
        plt.show()
        best_alpha = ccp_alphas[test_scores.index(max(test_scores))]
        final_clf_tree = DecisionTreeClassifier(ccp_alpha=best_alpha)
        final_clf_tree.fit(X_train, y_train)
        plt.figure(figsize=(12, 6))
        plot_tree(final_clf_tree, filled=True, feature_names=X.columns, class_names=label_encoder.classes_, fontsize=5)
        plt.title('Pruned Decision Tree', fontsize=12)
        plt.show()
        importances = clf3.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(X.shape[1]), importances[indices], align="center",color='green')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation='vertical')
        plt.title("Feature Importance - DecisionTree")
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.show()
        Dt_classification_report(y_test, y_pred, target_names=disease_options)
def Rf_classification_report(y_test,y_pred,target_names):
    report=classification_report(y_test,y_pred,target_names=target_names,output_dict=True)
    report_df=pd.DataFrame(report).transpose()
    report_df=report_df.round(2)
    print(report_df)
    plt.figure(figsize=(10,6))
    sns.heatmap(report_df.iloc[:-1,:].T,annot=True,cmap="Greens",fmt=".2f")
    plt.title('Classification Report--Random Forest')
    plt.show()
def randomforest():
    if len(NameEn.get()) == 0:
        pred2.set(" ")
        comp = messagebox.askokcancel("System", "Kindly fill the Name")
        if comp:
            root.mainloop()
    elif ((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")):
        pred2.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill At least first two symptoms")
        if sym:
            root.mainloop()
    else:
        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(),Symptom4.get()]
        if psymptoms.count("select here") < 4 and not check_duplicate_symptoms(psymptoms):
            messagebox.showerror("Input Error", "Please select distinct symptoms.")
            return
        from sklearn.ensemble import RandomForestClassifier
        clf4 = RandomForestClassifier(n_estimators=100)
        clf4 = clf4.fit(X, np.ravel(y))
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        y_pred = clf4.predict(X_test)
        acc2.set(str(accuracy_score(y_test, y_pred)))
        print("Random Forest")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))
        print("Confusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(),Symptom4.get()]
        for k in range(0, len(symptom_options)):
            for z in psymptoms:
                if (z == symptom_options[k]):
                    l2[k] = 1
        inputtest = [l2]
        predict = clf4.predict(inputtest)
        predicted = predict[0]
        h = 'no'
        for a in range(0, len(disease_options)):
            if predicted == a:
                h = 'yes'
                break
        if h == 'yes':
            pred2.set(" ")
            pred2.set(disease_options[a])
        else:
            pred2.set(" ")
            pred2.set("Not found")
            messagebox.showinfo("Prediction Result", "Disease not found")
        print("Selected Symptoms:", psymptoms)
        print("Test Set Predictions:", label_encoder.inverse_transform(y_pred))
        pred2.set(" ")
        pred2.set(disease_options[a])
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Greens', fmt='g',xticklabels=disease_options, yticklabels=disease_options)
        plt.title('Confusion Matrix - Random Forest')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        plt.figure(figsize=(10, 6))
        importances = clf4.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(X.shape[1]), importances[indices], align="center",color='green')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation='vertical')
        plt.title("Feature Importance - Random Forest")
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.show()
        Rf_classification_report(y_test, y_pred, target_names=disease_options)
def Svm_classification_report(y_test,y_pred,target_names):
    report=classification_report(y_test,y_pred,target_names=target_names,output_dict=True)
    report_df=pd.DataFrame(report).transpose()
    report_df=report_df.round(2)
    print(report_df)
    plt.figure(figsize=(10,6))
    sns.heatmap(report_df.iloc[:-1,:].T,annot=True,cmap="Oranges",fmt=".2f")
    plt.title('Classification Report--Support Vector Machine')
    plt.show()
from sklearn.multiclass import OneVsRestClassifier
def SVM():
    if len(NameEn.get()) == 0:
        pred3.set(" ")
        comp = messagebox.askokcancel("System", "Kindly fill the Name")
        if comp:
            root.mainloop()
    elif ((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")):
        pred3.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill At least first two symptoms")
        if sym:
            root.mainloop()
    else:
        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]
        if psymptoms.count("select here") < 4 and not check_duplicate_symptoms(psymptoms):
            messagebox.showerror("Input Error", "Please select distinct symptoms.")
            return
        clf_svm_ovr = OneVsRestClassifier(SVC(kernel='linear', probability=True))
        clf_svm_ovr.fit(X_train, y_train)
        y_pred = clf_svm_ovr.predict(X_test)
        y_pred_proba = clf_svm_ovr.predict_proba(X_test)
        acc3.set(str(accuracy_score(y_test, y_pred)))
        print("SVM with One-vs-Rest (OvR)")
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred, normalize=False))
        print("Confusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]
        for k in range(0, len(symptom_options)):
            for z in psymptoms:
                if z == symptom_options[k]:
                    l2[k] = 1
        inputtest = [l2]
        predict = clf_svm_ovr.predict(inputtest)
        predicted = predict[0]
        h = 'no'
        for a in range(0, len(disease_options)):
            if predicted == a:
                h = 'yes'
                break
        if h == 'yes':
            pred3.set(" ")
            pred3.set(disease_options[a])
        else:
            pred3.set(" ")
            pred3.set("Not found")
            messagebox.showinfo("Prediction Result", "Disease not found")
        print("Selected Symptoms:", psymptoms)
        print("Test Set Predictions:", label_encoder.inverse_transform(y_pred))
        pred3.set(" ")
        pred3.set(disease_options[a])
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Oranges', fmt='g', xticklabels=disease_options,yticklabels=disease_options)
        plt.title('Confusion Matrix - Support Vector Machine')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        Svm_classification_report(y_test, y_pred, target_names=disease_options)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
def AdaBoostWithHyperparameterTuning(X_train, X_test, y_train, y_test):
    messagebox.showinfo("Hyperparameter Tuning", "Please wait for the hyperparameter results.......")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1.0]
    }
    ada_clf = AdaBoostClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=ada_clf, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    messagebox.showinfo("Hyperparameter Tuning", "Classification Report After HyperTuning")
    Ada_classification_report(y_test, y_pred, target_names=disease_options)
    return best_params, best_model, accuracy
def plot_multiclass_roc(model, X_test, y_test, n_classes):
    y_score = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:0.2f})',color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:0.2f})',color='navy', linestyle=':', linewidth=4)
    colors = sns.color_palette("husl", n_classes)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,label=f'ROC curve of class {i} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multiclass')
    plt.legend(loc="lower right")
    plt.show()
def Ada_classification_report(y_test,y_pred,target_names):
    report=classification_report(y_test,y_pred,target_names=target_names,output_dict=True)
    report_df=pd.DataFrame(report).transpose()
    report_df=report_df.round(2)
    print(report_df)
    plt.figure(figsize=(10,6))
    sns.heatmap(report_df.iloc[:-1,:].T,annot=True,cmap="Purples",fmt=".2f")
    plt.title('Classification Report--AdaBoost')
    plt.show()
def AdaBoost():
    if len(NameEn.get()) == 0:
        pred4.set(" ")
        comp = messagebox.askokcancel("System", "Kindly fill the Name")
        if comp:
            root.mainloop()
    elif ((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")):
        pred4.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill At least first two symptoms")
        if sym:
            root.mainloop()
    else:
        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]
        if psymptoms.count("select here") < 4 and not check_duplicate_symptoms(psymptoms):
            messagebox.showerror("Input Error", "Please select distinct symptoms.")
            return
        accuracies = []
        n_estimators_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for n_estimators in n_estimators_list:
            clf_ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
            clf_ada.fit(X_train, y_train)
            y_pred = clf_ada.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators_list, accuracies, marker='o',color='green')
        plt.title('Accuracy vs. Number of Estimators Before Hyper Tuning - AdaBoost')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()
        best_n_estimators = n_estimators_list[np.argmax(accuracies)]
        print("Best number of estimators:", best_n_estimators)
        clf_ada_best = AdaBoostClassifier(n_estimators=best_n_estimators, random_state=42)
        clf_ada_best.fit(X_train, y_train)
        y_pred = clf_ada_best.predict(X_test)
        print("AdaBoost")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Purples', fmt='g',xticklabels=disease_options,yticklabels=disease_options)
        plt.title('Confusion Matrix - AdaBoost')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        plot_multiclass_roc(clf_ada_best, X_test, y_test, n_classes=len(label_encoder.classes_))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))
        best_params, best_model, accuracy = AdaBoostWithHyperparameterTuning(X_train, X_test, y_train, y_test)
        print("Best Parameters:", best_params)
        print("Test Accuracy:", accuracy)
        inputtest = [l2]
        predicted = clf_ada_best.predict(inputtest)
        pred4.set(disease_options[predicted[0]])
        acc4.set(str(accuracy))
        Ada_classification_report(y_test, y_pred, target_names=disease_options)
root.configure(background='black')
root.title('Smart Disease Predictor System')
root.attributes('-fullscreen', True)
root.resizable(False, False)
Symptom1 = StringVar()
Symptom1.set("Select Here")
Symptom2 = StringVar()
Symptom2.set("Select Here")
Symptom3 = StringVar()
Symptom3.set("Select Here")
Symptom4 = StringVar()
Symptom4.set("Select Here")
Name = StringVar()
prev_win = None
def Reset():
    global prev_win
    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    NameEn.delete(first=0,last=100)
    pred1.set(" ")
    pred2.set(" ")
    pred3.set(" ")
    pred4.set(" ")
    acc1.set(" ")
    acc2.set(" ")
    acc3.set(" ")
    acc4.set(" ")
    try:
        prev_win.destroy()
        prev_win = None
    except AttributeError:
        pass
from tkinter import messagebox
def Exit():
    qExit = messagebox.askyesno("System", "Do you want to exit the system")
    if qExit:
        root.destroy()
        exit()
w2 = Label(root, justify=CENTER, text="Goat and Sheep Disease Prediction", fg="#FF6347", bg="black")
w2.config(font=("Arial", 30, "bold italic"), bd=4, relief="solid")
w2.grid(row=0, column=0, columnspan=2, pady=50, sticky=E)
w3 = Label(root, justify=CENTER, text="Contributors: Prof.Kolluri Rajesh, Mr.Suluru Lokesh, Mr.Gunnam Chandramouli", fg="#32CD32", bg="black")
w3.config(font=("Arial", 30, "bold italic"), bd=4, relief="solid")
w3.grid(row=1, column=0, columnspan=5, pady=50, sticky=E)
NameLb = Label(root, text="Name of the patient *", fg="#4169E1", bg="white")
NameLb.config(font=("Arial", 15, "bold italic"), bd=4, relief="solid")
NameLb.grid(row=6, column=0, pady=(0, 0), sticky=W)
from tkinter import Canvas, font
def create_custom_label(root, text, row, column, shape_color, text_color, bg_color, font_style, pady, width):
    canvas = Canvas(root, width=width, height=40, bg=bg_color, highlightthickness=0)
    canvas.grid(row=row, column=column, pady=(0, pady), sticky='w')
    canvas.create_polygon(0, 20, 20, 0, width-20, 0, width, 20, width-20, 40, 20, 40, fill=shape_color)
    canvas.create_text(width/2, 20, text=text, font=font_style, fill=text_color)
create_custom_label(root, "Symptom 1 *", 7, 0, "#4169E1", "white", "Ivory", font.Font(family="Times", size=15, weight="bold", slant="italic"), 10, 200)
create_custom_label(root, "Symptom 2 *", 8, 0, "#008000", "white", "Ivory", font.Font(family="Times", size=15, weight="bold", slant="italic"), 10, 200)
create_custom_label(root, "Common Symptom 1", 9, 0, "#FF4500", "white", "Ivory", font.Font(family="Times", size=15, weight="bold", slant="italic"), 10, 250)
create_custom_label(root, "Common Symptom 2", 10, 0, "#FF4500", "white", "Ivory", font.Font(family="Times", size=16, weight="bold", slant="italic"), 10, 250)
lrLb = Label(root, text="Decision Tree", fg="White", bg="red", width=20)
lrLb.config(font=("Times", 15, "bold italic"))
lrLb.grid(row=15, column=0, pady=10, sticky=W)
acc_label1 = Label(root, text="Accuracy", font=("Times", 12, "bold italic")).grid(row=16, column=1)
acc_box1 = Label(root, font=("Times", 12, "normal"), textvariable=acc1, width=15, bg="lightblue", relief="sunken").grid(row=16, column=2)
destreeLb=Label(root,text="RandomForest",fg="Red",bg="Orange",width=20)
destreeLb.config(font=("Times",15,"bold italic"))
destreeLb.grid(row=17,column=0,pady=10,sticky=W)
acc_label2 = Label(root, text="Accuracy", font=("Times", 12, "bold italic")).grid(row=18, column=1)
acc_box2 = Label(root, font=("Times", 12, "normal"), textvariable=acc2, width=15, bg="lightblue", relief="sunken").grid(row=18, column=2)
svmLb = Label(root, text="SupportVectorMachine", fg="White", bg="green", width = 20)
svmLb.config(font=("Times",15,"bold italic"))
svmLb.grid(row=19, column=0, pady=10, sticky=W)
acc_label3 = Label(root, text="Accuracy", font=("Times", 12, "bold italic")).grid(row=20, column=1)
acc_box3 = Label(root, font=("Times", 12, "normal"), textvariable=acc3, width=15, bg="lightblue", relief="sunken").grid(row=20, column=2)
ada = Label(root, text="AdaBoost", fg="black", bg="gold", width=20)
ada.config(font=("Times", 15, "bold italic"))
ada.grid(row=21, column=0, pady=10, sticky=W)
ada = Label(root, text="AdaBoost", fg="black", bg="gold", width=20)
ada.config(font=("Times", 15, "bold italic"))
ada.grid(row=21, column=0, pady=10, sticky=W)
acc_label4 = Label(root, text="Accuracy", font=("Times", 12, "bold italic")).grid(row=22, column=1)
acc_box4 = Label(root, font=("Times", 12, "normal"), textvariable=acc4, width=15, bg="lightblue", relief="sunken").grid(row=22, column=2)
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)
option_menu_style = {'font': ('Arial', 12),'bg': 'white','fg': 'black','activebackground': 'lightgray','highlightthickness': 2,'highlightbackground': 'blue'}
s1 = OptionMenu(root, Symptom1, *symptom_options1)
s1.config(**option_menu_style)
s1.grid(row=7, column=1)
Symptom1.set("Select Here")
s2 = OptionMenu(root, Symptom2, *symptom_options1)
s2.config(**option_menu_style)
s2.grid(row=8, column=1)
Symptom2.set("Select Here")
s3 = OptionMenu(root, Symptom3, *common_symptoms_options)
s3.config(**option_menu_style)
s3.grid(row=9, column=1)
Symptom3.set("Select Here")
s4 = OptionMenu(root, Symptom4, *common_symptoms_options)
s4.config(**option_menu_style)
s4.grid(row=10, column=1)
Symptom4.set("Select Here")
button_style = {'font': ('Arial', 12),'bg': 'red','fg': 'yellow','activebackground': 'orange','highlightthickness': 2,'highlightbackground': 'black','relief': 'ridge'}
dst = Button(root, text="Prediction 1", command=DecisionTree)
dst.config(**button_style)
dst.grid(row=6, column=3, padx=10)
button_style_rnf = {'font': ('Arial', 12),'bg': 'light salmon','fg': 'blue','activebackground': 'salmon','highlightthickness': 2,'highlightbackground': 'black'}
rnf = Button(root, text="Prediction 2", command=randomforest)
rnf.config(**button_style_rnf)
rnf.grid(row=7, column=3, padx=10)
button_style_lr = {'font': ('Arial', 12),'bg': 'orange','fg': 'white','activebackground': 'darkorange','highlightthickness': 2,'highlightbackground': 'black'}
lr = Button(root, text="Prediction 3", command=SVM)
lr.config(**button_style_lr)
lr.grid(row=8, column=3, padx=10)
button_style_ada = {'font': ('Arial', 12),'bg': 'light yellow','fg': 'black','activebackground': 'yellow','highlightthickness': 2,'highlightbackground': 'black'}
ada = Button(root, text="Prediction 4", command=AdaBoost)
ada.config(**button_style_ada)
ada.grid(row=9, column=3, padx=10)
rs = Button(root,text="Reset Inputs", command=Reset,bg="yellow",fg="purple",width=15)
rs.config(font=("Times",15,"bold italic"))
rs.grid(row=11,column=3,padx=10)
ex = Button(root,text="Exit System", command=Exit,bg="yellow",fg="purple",width=15)
ex.config(font=("Times",15,"bold italic"))
ex.grid(row=12,column=3,padx=10)
t1 = Label(root, font=("Times", 15, "bold italic"), text='Decision Tree', height=1, bg='Light green',width=40, fg='red', textvariable=pred1, relief='sunken').grid(row=15, column=1, padx=10)
t2=Label(root,font=("Times",15,"bold italic"),text="Random Forest",height=1,bg="Purple",width=40,fg="white",textvariable=pred2,relief="sunken").grid(row=17, column=1, padx=10)
t3=Label(root,font=("Times",15,"bold italic"),text="SupportVectorMachine",height=1,bg="red",width=40,fg="orange",textvariable=pred3,relief="sunken").grid(row=19, column=1, padx=10)
t4 = Label(root, font=("Times", 15, "bold italic"), text="AdaBoost", height=1, bg="blue",width=40, fg="yellow", textvariable=pred4, relief="sunken").grid(row=21, column=1, padx=10, pady=10)
def final_prediction():
    predictions=[pred1.get(),pred2.get(),pred3.get(),pred4.get()]
    accuracies=[float(acc1.get()),float(acc2.get()),float(acc3.get()),float(acc4.get())]
    final_disease, final_accuracy = determine_final_disease(predictions,accuracies)
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, accuracies, marker='o', linestyle='-', color='green')
    plt.xlabel('Disease Prediction')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Disease Predictions')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    messagebox.showinfo("Final Prediction", f"Final Prediction: {final_disease}\nAccuracy: {final_accuracy}")
    display_final_disease_explanation(final_disease)
    models = ['Decision Tree', 'Random Forest', 'SupportVectorMachine','AdaBoost']
    accuracy = [accuracies]
    plt.figure(figsize=(8, 6))
    plt.plot(models, accuracies, marker='o', linestyle='-', color='orange')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies--Goat and Sheep Disease Prediction')
    plt.grid(True)
    plt.show()
    display_disease_image(final_disease)
fnd = Button(root, text="Final Prediction", command=final_prediction, bg="lightblue", fg="black")
fnd.config(font=("Times", 15, "bold italic"))
fnd.grid(row=10, column=3, padx=10)
def determine_final_disease(predictions,accuracies):
  highest_accuracy = max(accuracies)
  highest_accuracy_index=accuracies.index(highest_accuracy)
  final_disease=predictions[highest_accuracy_index]
  explanation = disease_explanations.get(final_disease, "No explanation available.")
  return final_disease,highest_accuracy
def display_final_disease_explanation(final_disease):
    explanation = disease_explanations.get(final_disease, "No explanation available.")
    explanation_window = Toplevel(root)
    explanation_window.title("Goat and Sheep Disease Explanation")
    explanation_window.geometry("500x300")
    explanation_label = Label(explanation_window, text=explanation, font=("Times", 12), wraplength=500)
    explanation_label.pack()
    close_button = Button(explanation_window, text="Close", command=explanation_window.destroy)
    close_button.pack()
def display_disease_image(predicted_disease):
    directory_path = r'C:\Users\User\OneDrive\Desktop\python_programs\sdp\Diseases_images'
    disease_images = {
        'Foot-Mouth-Disease(Goat)': 'foot.png',
        'Peste des petits ruminants(Goat)': 'pes.png',
        'BlueTongue(Goat)': 'blue.png',
        'Rift Valley fever(Goat)': 'rvf.png',
        'Border Disease(Goat)': 'borderdisease.png',
        'GoatPox(Goat)': 'goarpox.png',
        'ORF(Goat)': 'orfe.png',
        'Mastitis(Goat)': 'congested udder.png'
    }
    image_filename = disease_images.get(predicted_disease)
    if image_filename:
        image_path = fr'{directory_path}\{image_filename}'
        image_window = Toplevel()
        image_window.title(predicted_disease)
        image = PhotoImage(file=image_path)
        image_label = Label(image_window, image=image)
        image_label.image = image
        image_label.pack()
from sklearn.feature_selection import SelectFromModel
def display_top_symptoms(clf,selected_symptoms):
    if hasattr(clf,'feature_importances_'):
        feature_importances=clf.feature_importances_
        selected_features_indices=feature_importances.nonzero()[0]
        selected_feature_names=[selected_symptoms[i] for i in selected_features_indices]
        selected_features_importances=[feature_importances[i] for i in selected_features_indices]
        sorted_indices=sorted(range(len(selected_features_importances)),key=lambda i:  selected_features_importances[i],reverse=True)
        print("Top Symptoms Contributing to Predicted Disease:")
        for i in sorted_indices:
            print(f"{selected_feature_names[i]}:{selected_features_importances[i]}")
root.mainloop()